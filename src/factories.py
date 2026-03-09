DATASETS = {
    'FinanceInc/auditor_sentiment': {
        'text_column': 'sentence',
        'label_column': 'label',
        'subset': None,
        'validation_key': 'test',
        'description': 'The dataset consists of sentences from English language of auditor sentiments from financial news.',
        'target_variable': 'sentiment label (negative, neutral, positive)',
        'possible_classes': ['negative', 'neutral', 'positive'],
        'class_map': {
            0: 'negative',
            1: 'neutral',
            2: 'positive',
        },
    },
    'ugursa/Yahoo-Finance-News-Sentences': {
        'text_column': 'text',
        'label_column': 'label',
        'subset': None,
        'validation_key': None,
        'description': 'The dataset consists of sentences from English language of sentiments from financial news.',
        'target_variable': 'sentiment label (positive, neutral, negative)',
        'possible_classes': ['positive', 'neutral', 'negative'],
        'class_map': {
            0: 'positive',
            1: 'neutral',
            2: 'negative',
        },
    },
    'zeroshot/twitter-financial-news-sentiment': {
        'text_column': 'text',
        'label_column': 'label',
        'subset': None,
        'validation_key': 'validation',
        'description': '.',
        'target_variable': 'sentiment label (bearish, bullish, neutral)',
        'possible_classes': ['bearish', 'bullish', 'neutral'],
        'class_map': {
            0: 'bearish',
            1: 'bullish',
            2: 'neutral',
        },
    },
    'lmsys/toxic-chat': {
        'text_column': 'text',
        'label_column': 'toxicity',
        'subset': 'toxicchat0124',
        'validation_key': 'test',
        'description': 'The dataset contains prompts and responses labeled for toxicity.',
        'target_variable': 'toxicity label (non-toxic, toxic)',
        'possible_classes': ['non-toxic', 'toxic'],
        'class_map': {
            0: 'non-toxic',
            1: 'toxic',
        },
    },
}

import optimizers

OPTIMIZERS = {
    'example-mipro': optimizers.PerExampleMIPROOptimizer,
    #####################################################
    'bag-mipro': optimizers.PerBagMIPROOptimizer,
    'bag-mipro-feedback': optimizers.PerBagMIPROWithFeedbackOptimizer,
}

import evaluators

EVALUATORS = {
    'example-correctness': evaluators.PerExampleCorrectnessScorer,
    'example-correctness-interpretability': evaluators.PerExampleCorrectnessAndInterpretabilityScorer,
    #########################################################
    'bag-correctness': evaluators.PerBagCorrectnessScorer,
    'bag-correctness-interpretability': evaluators.PerBagCorrectnessAndInterpretabilityScorer,
    'bag-correctness-interpretability-feedback': evaluators.PerBagCorrectnessAndInterpretabilityWithFeedbackScorer,
}
