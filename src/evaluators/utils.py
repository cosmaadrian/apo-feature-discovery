import shap
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif


def train_lr_classifier(df, feature_types):
    feature_names = [c for c in df.columns if c.startswith('feat:')]
    categorical_features = [f'feat:{name}' for name, t in feature_types.items() if t in ['Literal', 'bool']]
    numeric_features = [f'feat:{name}' for name, t in feature_types.items() if t in ['int', 'float']]

    X = df.drop(columns = ['text', 'label'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, stratify = y, test_size = 0.2)

    # cast bools to category to avoid numeric-sorting rule
    df[categorical_features] = df[categorical_features].apply(lambda s: s.astype('category'))
    enc = OneHotEncoder(categories = [list(s.cat.categories) for _, s in df[categorical_features].items()], handle_unknown = 'ignore', drop = 'if_binary', sparse_output = False)

    preprocessor = ColumnTransformer(
        transformers = [
            ('numeric', StandardScaler(with_mean = False), numeric_features),
            ('categorical', enc, categorical_features),
        ],
        remainder = 'drop',
        sparse_threshold = 0.0,
    )

    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    clf = LogisticRegression(max_iter = 100)
    clf.fit(X_train_scaled, y_train)

    f1_macro = f1_score(y_test, clf.predict(X_test_scaled), average = 'macro')
    f1_micro = f1_score(y_test, clf.predict(X_test_scaled), average = 'micro')

    prec_macro = precision_score(y_test, clf.predict(X_test_scaled), average = 'macro')
    prec_micro = precision_score(y_test, clf.predict(X_test_scaled), average = 'micro')

    rec_macro = recall_score(y_test, clf.predict(X_test_scaled), average = 'macro')
    rec_micro = recall_score(y_test, clf.predict(X_test_scaled), average = 'micro')

    # --- build transformed feature names and mapping back to original features ---
    transformed_names = []
    original_of_transformed = []

    # numeric keep names
    transformed_names.extend(numeric_features)
    original_of_transformed.extend(numeric_features)

    # categorical expand robustly using encoder.categories_ aligned to categorical_features
    ohe = preprocessor.named_transformers_['categorical']

    if ohe.categories:
        for f, cats, drop_idx in zip(categorical_features, ohe.categories, ohe.drop_idx_):
            kept = [c for j, c in enumerate(cats) if (drop_idx is None or j != drop_idx)]
            for cat in kept:
                if len(kept) == 1:
                    transformed_names.append(f)
                else:
                    transformed_names.append(f"{f}:{cat}")

                original_of_transformed.append(f)

    aggregated = {f: 0.0 for f in feature_names}

    # Feature coverage
    coverage = {}
    for feat in feature_names:
        if feat in categorical_features:
            # compute normalized shannon entropy
            counts = df[feat].value_counts(normalize = True)
            shannon_entropy = -np.sum(counts * np.log2(counts + 1e-12))
            max_entropy = np.log2(len(counts))
            coverage[feat] = shannon_entropy / max_entropy if max_entropy > 0 else 0.0
        elif feat in numeric_features:
            # compute distributional coverage
            x = df[feat].to_numpy()
            x = x[np.isfinite(x)]
            hist, _ = np.histogram(x, bins = 10, density = True)
            p = hist / hist.sum()
            entropy = -(p * np.log(p + 1e-12)).sum() / np.log(len(p))
            coverage[feat] = entropy

    # SHAP importance
    explainer = shap.LinearExplainer(clf, X_train_scaled)
    shap_values = explainer.shap_values(X_test_scaled)

    if isinstance(shap_values, list):
        sv = np.mean([np.abs(sv) for sv in shap_values], axis = 0)
    else:
        sv = np.abs(shap_values)

    global_imp_transformed = sv.mean(axis = 0)

    # aggregate transformed -> original
    for imp, orig in zip(global_imp_transformed, original_of_transformed):
        if isinstance(imp, np.ndarray):
            aggregated[orig] += float(np.max(imp))
        else:
            aggregated[orig] += imp

    # Mutual Information
    mi = mutual_info_classif(X_train_scaled, y_train, discrete_features = 'auto')

    # handle categorical features with multiple one-hot columns by summing their mi scores
    aggregated_mi = {f: 0.0 for f in feature_names}
    for score, orig in zip(mi, original_of_transformed):
        aggregated_mi[orig] += score

    global_feature_importances = []
    for feat in feature_names:
        global_feature_importances.append({'feature_name': feat.replace('feat:', ''), 'importance': round(float(aggregated[feat]), 3), 'mi': round(float(aggregated_mi[feat]), 3), 'coverage': round(float(coverage[feat]), 3)})

    return {
        'metrics': {
            'f1:macro': f1_macro,
            'f1:micro': f1_micro,
            'precision:macro': prec_macro,
            'precision:micro': prec_micro,
            'recall:macro': rec_macro,
            'recall:micro': rec_micro,
        },
        'feature_importances': global_feature_importances,
    }
