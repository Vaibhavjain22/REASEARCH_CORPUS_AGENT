

TEST_QUERIES = [

    # ─────────────────────────────────────────────────
    # SIMPLE QUERIES (3)
    # ─────────────────────────────────────────────────
    {
        "query": "What is dynamic backtracking in search algorithms?",
        "relevant_keywords": ["backtracking", "search", "backtrack", "depth"],
        "type": "simple"
    },
    {
        "query": "Explain reinforcement learning with temporal difference methods",
        "relevant_keywords": ["reinforcement", "temporal", "learning", "reward"],
        "type": "simple"
    },
    {
        "query": "What is word sense disambiguation in natural language processing?",
        "relevant_keywords": ["word", "sense", "disambiguation", "language"],
        "type": "simple"
    },
    {
        "query": "Explain support vector machines for classification",
        "relevant_keywords": ["support", "vector", "classifier", "kernel"],
        "type": "simple"
    },
    {
        "query": "What is naive bayes classification method?",
        "relevant_keywords": ["naive", "bayes", "classification", "probability"],
        "type": "simple"
    },

    # ─────────────────────────────────────────────────
    # MULTI-HOP QUERIES (5)
    # ─────────────────────────────────────────────────
    {
        "query": "How does reinforcement learning apply to game tree search "
                 "and what role does temporal difference learning play?",
        "relevant_keywords": ["reinforcement", "game", "temporal", "search"],
        "type": "multi_hop"
    },
    {
        "query": "How do neural networks combined with evolutionary computing "
                 "improve optimization problems?",
        "relevant_keywords": ["neural", "evolutionary", "optimization", "network"],
        "type": "multi_hop"
    },
    {
        "query": "How does minimum description length principle apply to "
                 "substructure discovery and knowledge representation?",
        "relevant_keywords": ["description", "length", "knowledge", "learning"],
        "type": "multi_hop"
    },
    {
        "query": "How do boosting methods improve word sense disambiguation "
                 "and text classification tasks?",
        "relevant_keywords": ["boosting", "word", "classification", "text"],
        "type": "multi_hop"
    },
    {
        "query": "How does dimensionality reduction using manifold learning "
                 "help in computer vision pattern recognition?",
        "relevant_keywords": ["dimension", "manifold", "vision", "pattern"],
        "type": "multi_hop"
    },

    # ─────────────────────────────────────────────────
    # COMPARISON QUERIES (5)
    # ─────────────────────────────────────────────────
    {
        "query": "Compare total-order planning and partial-order planning "
                 "approaches in AI",
        "relevant_keywords": ["planning", "partial", "order", "total"],
        "type": "comparison"
    },
    {
        "query": "Compare naive bayes and exemplar based approaches "
                 "to word sense disambiguation",
        "relevant_keywords": ["naive", "bayes", "exemplar", "disambiguation"],
        "type": "comparison"
    },
    {
        "query": "Compare symbolic pattern associator and connectionist "
                 "models for language learning",
        "relevant_keywords": ["symbolic", "connectionist", "learning", "model"],
        "type": "comparison"
    },
    {
        "query": "Compare decision tree induction methods and their "
                 "complexity tradeoffs",
        "relevant_keywords": ["decision", "tree", "induction", "complexity"],
        "type": "comparison"
    },
    {
        "query": "Compare linear function approximation and nonlinear "
                 "methods in reinforcement learning",
        "relevant_keywords": ["linear", "reinforcement", "approximation", "learning"],
        "type": "comparison"
    },

    # ─────────────────────────────────────────────────
    # AGGREGATION QUERIES (5)
    # ─────────────────────────────────────────────────
    {
        "query": "What are the common evaluation methods used across "
                 "machine learning research papers?",
        "relevant_keywords": ["evaluation", "method", "learning", "empirical"],
        "type": "aggregation"
    },
    {
        "query": "What optimization techniques are most commonly used "
                 "in computer vision papers?",
        "relevant_keywords": ["optimization", "vision", "method", "algorithm"],
        "type": "aggregation"
    },
    {
        "query": "What are the main challenges reported in natural language "
                 "processing research?",
        "relevant_keywords": ["language", "processing", "challenge", "model"],
        "type": "aggregation"
    },
    {
        "query": "What knowledge representation techniques are used "
                 "in artificial intelligence systems?",
        "relevant_keywords": ["knowledge", "representation", "reasoning", "logic"],
        "type": "aggregation"
    },
    {
        "query": "What are the most used learning algorithms reported "
                 "across neural and evolutionary computing papers?",
        "relevant_keywords": ["learning", "neural", "algorithm", "evolutionary"],
        "type": "aggregation"
    },

    # ─────────────────────────────────────────────────
    # FAILURE / EDGE CASES (5)
    # ─────────────────────────────────────────────────
    {
        "query": "What is the best recipe for biryani at home?",
        "relevant_keywords": ["recipe", "biryani", "food", "cooking"],
        "type": "failure_case"
    },
    {
        "query": "Who won the IPL cricket tournament in 2023?",
        "relevant_keywords": ["IPL", "cricket", "tournament", "winner"],
        "type": "failure_case"
    },
    {
        "query": "What are the latest smartphone models released in 2024?",
        "relevant_keywords": ["smartphone", "mobile", "phone", "release"],
        "type": "failure_case"
    },
    {
        "query": "xkqzpwm jfhvbnr tlsydoq random nonsense query",
        "relevant_keywords": ["nonsense", "random", "gibberish"],
        "type": "failure_case"
    },
    {
        "query": "What is the current stock price of Tesla today?",
        "relevant_keywords": ["stock", "Tesla", "price", "market"],
        "type": "failure_case"
    },
]
