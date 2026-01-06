"""3-stage LLM Council orchestration."""

from typing import List, Dict, Any, Tuple
from .openrouter import query_models_parallel, query_model
from .config import COUNCIL_MODELS, CHAIRMAN_MODEL


async def stage1_collect_responses(user_query: str) -> List[Dict[str, Any]]:
    """
    Stage 1: Collect individual responses from all council models.

    Args:
        user_query: The user's question

    Returns:
        List of dicts with 'model' and 'response' keys
    """
    messages = [{"role": "user", "content": user_query}]

    # Query all models in parallel
    responses = await query_models_parallel(COUNCIL_MODELS, messages)

    # Format results
    stage1_results = []
    for model, response in responses.items():
        if response is not None:  # Only include successful responses
            stage1_results.append({
                "model": model,
                "response": response.get('content', '')
            })

    return stage1_results


async def stage2_collect_rankings(
    user_query: str,
    stage1_results: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Stage 2: Each model ranks the anonymized responses.

    Args:
        user_query: The original user query
        stage1_results: Results from Stage 1

    Returns:
        Tuple of (rankings list, label_to_model mapping)
    """
    # Create anonymized labels for responses (Response A, Response B, etc.)
    labels = [chr(65 + i) for i in range(len(stage1_results))]  # A, B, C, ...

    # Create mapping from label to model name
    label_to_model = {
        f"Response {label}": result['model']
        for label, result in zip(labels, stage1_results)
    }

    # Build the ranking prompt
    responses_text = "\n\n".join([
        f"Response {label}:\n{result['response']}"
        for label, result in zip(labels, stage1_results)
    ])

    ranking_prompt = f"""You are evaluating different responses to the following question:

Question: {user_query}

Here are the responses from different models (anonymized):

{responses_text}

Your task:
1. First, evaluate each response individually. For each response, explain what it does well and what it does poorly.
2. Then, at the very end of your response, provide a final ranking.

IMPORTANT: Your final ranking MUST be formatted EXACTLY as follows:
- Start with the line "FINAL RANKING:" (all caps, with colon)
- Then list the responses from best to worst as a numbered list
- Each line should be: number, period, space, then ONLY the response label (e.g., "1. Response A")
- Do not add any other text or explanations in the ranking section

Example of the correct format for your ENTIRE response:

Response A provides good detail on X but misses Y...
Response B is accurate but lacks depth on Z...
Response C offers the most comprehensive answer...

FINAL RANKING:
1. Response C
2. Response A
3. Response B

Now provide your evaluation and ranking:"""

    messages = [{"role": "user", "content": ranking_prompt}]

    # Get rankings from all council models in parallel
    responses = await query_models_parallel(COUNCIL_MODELS, messages)

    # Format results
    stage2_results = []
    for model, response in responses.items():
        if response is not None:
            full_text = response.get('content', '')
            parsed = parse_ranking_from_text(full_text)
            stage2_results.append({
                "model": model,
                "ranking": full_text,
                "parsed_ranking": parsed
            })

    return stage2_results, label_to_model


async def stage3_synthesize_final(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    stage2_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Stage 3: Chairman synthesizes final response.

    Args:
        user_query: The original user query
        stage1_results: Individual model responses from Stage 1
        stage2_results: Rankings from Stage 2

    Returns:
        Dict with 'model' and 'response' keys
    """
    # Build comprehensive context for chairman
    stage1_text = "\n\n".join([
        f"Model: {result['model']}\nResponse: {result['response']}"
        for result in stage1_results
    ])

    stage2_text = "\n\n".join([
        f"Model: {result['model']}\nRanking: {result['ranking']}"
        for result in stage2_results
    ])

    chairman_prompt = f"""You are the Chairman of an LLM Council. Multiple AI models have provided responses to a user's question, and then ranked each other's responses.

Original Question: {user_query}

STAGE 1 - Individual Responses:
{stage1_text}

STAGE 2 - Peer Rankings:
{stage2_text}

Your task as Chairman is to synthesize all of this information into a single, comprehensive, accurate answer to the user's original question. Consider:
- The individual responses and their insights
- The peer rankings and what they reveal about response quality
- Any patterns of agreement or disagreement

Provide a clear, well-reasoned final answer that represents the council's collective wisdom:"""

    messages = [{"role": "user", "content": chairman_prompt}]

    # Query the chairman model
    response = await query_model(CHAIRMAN_MODEL, messages)

    if response is None:
        # Fallback if chairman fails
        return {
            "model": CHAIRMAN_MODEL,
            "response": "Error: Unable to generate final synthesis."
        }

    return {
        "model": CHAIRMAN_MODEL,
        "response": response.get('content', '')
    }


def parse_ranking_from_text(ranking_text: str) -> List[str]:
    """
    Parse the FINAL RANKING section from the model's response.

    Args:
        ranking_text: The full text response from the model

    Returns:
        List of response labels in ranked order
    """
    import re

    # Look for "FINAL RANKING:" section
    if "FINAL RANKING:" in ranking_text:
        # Extract everything after "FINAL RANKING:"
        parts = ranking_text.split("FINAL RANKING:")
        if len(parts) >= 2:
            ranking_section = parts[1]
            # Try to extract numbered list format (e.g., "1. Response A")
            # This pattern looks for: number, period, optional space, "Response X"
            numbered_matches = re.findall(r'\d+\.\s*Response [A-Z]', ranking_section)
            if numbered_matches:
                # Extract just the "Response X" part
                return [re.search(r'Response [A-Z]', m).group() for m in numbered_matches]

            # Fallback: Extract all "Response X" patterns in order
            matches = re.findall(r'Response [A-Z]', ranking_section)
            return matches

    # Fallback: try to find any "Response X" patterns in order
    matches = re.findall(r'Response [A-Z]', ranking_text)
    return matches


def calculate_aggregate_rankings(
    stage2_results: List[Dict[str, Any]],
    label_to_model: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    Calculate aggregate rankings across all models.

    Args:
        stage2_results: Rankings from each model
        label_to_model: Mapping from anonymous labels to model names

    Returns:
        List of dicts with model name and average rank, sorted best to worst
    """
    from collections import defaultdict

    # Track positions for each model
    model_positions = defaultdict(list)

    for ranking in stage2_results:
        ranking_text = ranking['ranking']

        # Parse the ranking from the structured format
        parsed_ranking = parse_ranking_from_text(ranking_text)

        for position, label in enumerate(parsed_ranking, start=1):
            if label in label_to_model:
                model_name = label_to_model[label]
                model_positions[model_name].append(position)

    # Calculate average position for each model
    aggregate = []
    for model, positions in model_positions.items():
        if positions:
            avg_rank = sum(positions) / len(positions)
            aggregate.append({
                "model": model,
                "average_rank": round(avg_rank, 2),
                "rankings_count": len(positions)
            })

    # Sort by average rank (lower is better)
    aggregate.sort(key=lambda x: x['average_rank'])

    return aggregate


def calculate_tournament_rankings(
    stage2_results: List[Dict[str, Any]],
    label_to_model: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    Calculate rankings using tournament-style pairwise comparison.

    For each pair of models, count how many rankers preferred one over the other.
    The model with more pairwise wins ranks higher. This method is more robust
    to outlier rankings than simple position averaging.

    Args:
        stage2_results: Rankings from each model with parsed_ranking
        label_to_model: Mapping from anonymous labels to model names

    Returns:
        List of dicts sorted by win_percentage (descending):
        [
            {
                "model": "openai/gpt-4o",
                "wins": 4.0,
                "losses": 1.0,
                "ties": 1.0,
                "win_percentage": 0.75,
                "total_matchups": 6
            },
            ...
        ]
    """
    from collections import defaultdict

    # Get all models from label_to_model
    models = list(set(label_to_model.values()))

    if len(models) < 2:
        # Need at least 2 models for pairwise comparison
        return [{"model": m, "wins": 0, "losses": 0, "ties": 0, "win_percentage": 0.0, "total_matchups": 0} for m in models]

    # Track pairwise wins: pairwise_wins[(model_a, model_b)] = count of times a ranked above b
    pairwise_wins = defaultdict(int)

    # Process each ranker's parsed ranking
    # Use pre-parsed ranking if available, otherwise parse from text
    for ranking in stage2_results:
        parsed_ranking = ranking.get('parsed_ranking')
        if not parsed_ranking:
            # Fallback: parse from raw ranking text (consistent with calculate_aggregate_rankings)
            ranking_text = ranking.get('ranking', '')
            parsed_ranking = parse_ranking_from_text(ranking_text) if ranking_text else []

        if not parsed_ranking:
            continue

        # Convert labels to model names and get their positions
        model_positions = {}
        for position, label in enumerate(parsed_ranking):
            if label in label_to_model:
                model_name = label_to_model[label]
                model_positions[model_name] = position

        # For each pair of models, record who was ranked higher (lower position = better)
        ranked_models = list(model_positions.keys())
        for i in range(len(ranked_models)):
            for j in range(i + 1, len(ranked_models)):
                model_a = ranked_models[i]
                model_b = ranked_models[j]
                pos_a = model_positions[model_a]
                pos_b = model_positions[model_b]

                # Ensure consistent ordering for the key
                if model_a > model_b:
                    model_a, model_b = model_b, model_a
                    pos_a, pos_b = pos_b, pos_a

                if pos_a < pos_b:
                    pairwise_wins[(model_a, model_b, 'a')] += 1
                elif pos_b < pos_a:
                    pairwise_wins[(model_a, model_b, 'b')] += 1
                # Equal positions would be a tie (shouldn't happen with rankings)

    # Calculate wins, losses, and ties for each model
    model_stats = {model: {"wins": 0.0, "losses": 0.0, "ties": 0.0} for model in models}

    # Process each unique pair of models
    processed_pairs = set()
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            model_a, model_b = models[i], models[j]
            if model_a > model_b:
                model_a, model_b = model_b, model_a

            pair_key = (model_a, model_b)
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)

            a_wins = pairwise_wins.get((model_a, model_b, 'a'), 0)
            b_wins = pairwise_wins.get((model_a, model_b, 'b'), 0)

            if a_wins > b_wins:
                model_stats[model_a]["wins"] += 1
                model_stats[model_b]["losses"] += 1
            elif b_wins > a_wins:
                model_stats[model_b]["wins"] += 1
                model_stats[model_a]["losses"] += 1
            elif a_wins == b_wins and (a_wins > 0 or b_wins > 0):
                # Tie - both get 0.5
                model_stats[model_a]["ties"] += 1
                model_stats[model_b]["ties"] += 1

    # Calculate win percentage and build results
    total_possible_matchups = len(models) - 1 if len(models) > 1 else 1
    results = []

    for model in models:
        stats = model_stats[model]
        total_matchups = stats["wins"] + stats["losses"] + stats["ties"]
        # Win percentage: wins + 0.5*ties / total matchups
        if total_matchups > 0:
            win_pct = (stats["wins"] + 0.5 * stats["ties"]) / total_possible_matchups
        else:
            win_pct = 0.0

        results.append({
            "model": model,
            "wins": stats["wins"],
            "losses": stats["losses"],
            "ties": stats["ties"],
            "win_percentage": round(win_pct, 3),
            "total_matchups": int(total_matchups)
        })

    # Sort by win percentage (higher is better)
    results.sort(key=lambda x: (-x['win_percentage'], x['losses']))

    return results


def detect_minority_opinions(
    stage2_results: List[Dict[str, Any]],
    label_to_model: Dict[str, str],
    tournament_rankings: List[Dict[str, Any]],
    dissent_threshold: float = 0.3,
    position_tolerance: int = 1
) -> List[Dict[str, Any]]:
    """
    Detect minority opinions where a significant portion of rankers disagree
    with the consensus ranking for a specific model.

    A minority opinion is flagged when ≥dissent_threshold of rankers place a model
    more than position_tolerance positions away from its consensus position.

    Args:
        stage2_results: Rankings from each model with parsed_ranking
        label_to_model: Mapping from anonymous labels to model names
        tournament_rankings: Consensus ranking from tournament method
        dissent_threshold: Minimum fraction of rankers that must disagree (default 0.3 = 30%)
        position_tolerance: How many positions away counts as disagreement (default 1)

    Returns:
        List of minority opinion dicts:
        [
            {
                "model": "openai/gpt-4o",
                "consensus_position": 1,
                "dissent_positions": [3, 4],  # where dissenters placed it
                "dissent_rate": 0.4,
                "dissenters": ["anthropic/claude-3.5-sonnet", "google/gemini-2.0-flash"],
                "direction": "undervalued"  # or "overvalued" - dissenters think it's worse/better
            },
            ...
        ]
    """
    from collections import defaultdict

    if not stage2_results or not tournament_rankings:
        return []

    # Build consensus position lookup from tournament rankings
    consensus_positions = {
        entry["model"]: position + 1  # 1-indexed
        for position, entry in enumerate(tournament_rankings)
    }

    # Track each ranker's position for each model
    # Structure: {model_name: [(ranker_model, position), ...]}
    model_rankings_by_ranker = defaultdict(list)

    for ranking in stage2_results:
        ranker_model = ranking.get('model')
        parsed_ranking = ranking.get('parsed_ranking')
        if not parsed_ranking:
            ranking_text = ranking.get('ranking', '')
            parsed_ranking = parse_ranking_from_text(ranking_text) if ranking_text else []

        if not parsed_ranking:
            continue

        # Record where this ranker placed each model
        for position, label in enumerate(parsed_ranking, start=1):
            if label in label_to_model:
                model_name = label_to_model[label]
                model_rankings_by_ranker[model_name].append((ranker_model, position))

    # Detect minority opinions for each model
    minority_opinions = []

    for model_name, rankings in model_rankings_by_ranker.items():
        if model_name not in consensus_positions:
            continue

        consensus_pos = consensus_positions[model_name]
        total_rankers = len(rankings)

        if total_rankers == 0:
            continue

        # Find dissenters: rankers who placed this model far from consensus
        dissenters = []
        dissent_positions = []

        for ranker_model, ranker_position in rankings:
            position_diff = abs(ranker_position - consensus_pos)
            if position_diff > position_tolerance:
                dissenters.append(ranker_model)
                dissent_positions.append(ranker_position)

        dissent_rate = len(dissenters) / total_rankers

        # Only report if dissent rate meets threshold
        if dissent_rate >= dissent_threshold and dissenters:
            # Determine direction: are dissenters ranking it higher or lower?
            avg_dissent_pos = sum(dissent_positions) / len(dissent_positions)
            if avg_dissent_pos > consensus_pos:
                direction = "overvalued"  # consensus ranks it higher than dissenters think
            else:
                direction = "undervalued"  # consensus ranks it lower than dissenters think

            minority_opinions.append({
                "model": model_name,
                "consensus_position": consensus_pos,
                "dissent_positions": sorted(set(dissent_positions)),
                "dissent_rate": round(dissent_rate, 2),
                "dissenters": dissenters,
                "direction": direction
            })

    # Sort by dissent rate (highest first)
    minority_opinions.sort(key=lambda x: -x['dissent_rate'])

    return minority_opinions


async def generate_conversation_title(user_query: str) -> str:
    """
    Generate a short title for a conversation based on the first user message.

    Args:
        user_query: The first user message

    Returns:
        A short title (3-5 words)
    """
    title_prompt = f"""Generate a very short title (3-5 words maximum) that summarizes the following question.
The title should be concise and descriptive. Do not use quotes or punctuation in the title.

Question: {user_query}

Title:"""

    messages = [{"role": "user", "content": title_prompt}]

    # Use gemini-2.5-flash for title generation (fast and cheap)
    response = await query_model("google/gemini-2.5-flash", messages, timeout=30.0)

    if response is None:
        # Fallback to a generic title
        return "New Conversation"

    title = response.get('content', 'New Conversation').strip()

    # Clean up the title - remove quotes, limit length
    title = title.strip('"\'')

    # Truncate if too long
    if len(title) > 50:
        title = title[:47] + "..."

    return title


async def run_full_council(user_query: str) -> Tuple[List, List, Dict, Dict]:
    """
    Run the complete 3-stage council process.

    Args:
        user_query: The user's question

    Returns:
        Tuple of (stage1_results, stage2_results, stage3_result, metadata)
    """
    # Stage 1: Collect individual responses
    stage1_results = await stage1_collect_responses(user_query)

    # If no models responded successfully, return error
    if not stage1_results:
        return [], [], {
            "model": "error",
            "response": "All models failed to respond. Please try again."
        }, {}

    # Stage 2: Collect rankings
    stage2_results, label_to_model = await stage2_collect_rankings(user_query, stage1_results)

    # Calculate aggregate rankings (both methods)
    aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)
    tournament_rankings = calculate_tournament_rankings(stage2_results, label_to_model)

    # Detect minority opinions
    minority_opinions = detect_minority_opinions(
        stage2_results, label_to_model, tournament_rankings
    )

    # Stage 3: Synthesize final answer
    stage3_result = await stage3_synthesize_final(
        user_query,
        stage1_results,
        stage2_results
    )

    # Prepare metadata
    metadata = {
        "label_to_model": label_to_model,
        "aggregate_rankings": aggregate_rankings,
        "tournament_rankings": tournament_rankings,
        "minority_opinions": minority_opinions
    }

    return stage1_results, stage2_results, stage3_result, metadata
