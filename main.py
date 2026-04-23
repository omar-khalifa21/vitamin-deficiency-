import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from sklearn.feature_selection import mutual_info_regression

from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
TARGET_COLUMN = "vitamin_deficiency"
DATA_PATH = "train_data_processed.csv"
OUTPUT_DIR = ".\\outputs"

SYMPTOM_COLUMNS = [
    "night_blindness",
    "dry_skin",
    "memory_problems",
    "pale_skin",
    "numbness_tingling",
    "bone_pain",
    "fatigue",
    "muscle_weakness",
    "bleeding_gums",
]

NUTRIENT_COLUMNS = [
    "vitamin_a_percent_rda",
    "vitamin_c_percent_rda",
    "vitamin_d_percent_rda",
    "vitamin_e_percent_rda",
    "vitamin_b12_percent_rda",
    "folate_percent_rda",
    "calcium_percent_rda",
    "iron_percent_rda",
]




def engineer_features(df: pd.DataFrame):
    data = df.copy()


    data = data.drop(columns=["symptoms_count"])


    data["symptom_neuro_score"] = data[["memory_problems", "numbness_tingling", "muscle_weakness"]].sum(axis=1)

    data["symptom_skin_score"] = data[["night_blindness", "dry_skin", "bleeding_gums"]].sum(axis=1)
    data["symptom_energy_score"] = data[["fatigue", "pale_skin", "bone_pain"]].sum(axis=1)

    data["nutrient_mean"] = data[NUTRIENT_COLUMNS].mean(axis=1)
    data["nutrient_min"] = data[NUTRIENT_COLUMNS].min(axis=1)
    data["fat_soluble_mean"] = data[["vitamin_a_percent_rda", "vitamin_d_percent_rda", "vitamin_e_percent_rda"]].mean(axis=1)
    data["blood_health_mean"] = data[["vitamin_b12_percent_rda", "folate_percent_rda", "iron_percent_rda"]].mean(axis=1)
    data["bone_health_mean"] = data[["vitamin_d_percent_rda", "calcium_percent_rda"]].mean(axis=1)
    data["lifestyle_risk_score"] = (data["smoking_status"] + data["alcohol_consumption"] - data["exercise_level"])
    data["sun_d_interaction"] = data["sun_exposure"] * data["vitamin_d_percent_rda"]
    data["bmi_age_interaction"] = data["bmi"] * data["age"]
    data["income_diet_interaction"] = data["income_level"] * data["diet_type"]

    return data


def remove_unwanted_features(X: pd.DataFrame, correlation_threshold: float = 0.95):
    cleaned = X.copy()
    dropped_features = []

    # duplicate_rows = cleaned.duplicated().sum()
    # if duplicate_rows:
    #     cleaned = cleaned.loc[~cleaned.duplicated()].copy()

    zero_variance = [column for column in cleaned.columns if cleaned[column].nunique() <= 1]
    if zero_variance:
        cleaned = cleaned.drop(columns=zero_variance)
        dropped_features.extend(zero_variance)

    correlation_matrix = cleaned.corr(numeric_only=True).abs()
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    highly_correlated = [column for column in upper_triangle.columns if any(upper_triangle[column] > correlation_threshold)]

    if highly_correlated:
        cleaned = cleaned.drop(columns=highly_correlated)
        dropped_features.extend(highly_correlated)

    return cleaned, dropped_features


def select_features(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, top_k: int = 18):
    top_k = min(top_k, X_train.shape[1])
    scores = mutual_info_regression(X_train, y_train, random_state=RANDOM_STATE)
    feature_scores = pd.Series(scores, index=X_train.columns).sort_values(ascending=False)
    selected_columns = feature_scores.head(top_k).index.tolist()
    return X_train[selected_columns], X_test[selected_columns], feature_scores


def detect_outliers_iqr(X: pd.DataFrame):
    numeric_columns = X.select_dtypes(include=[np.number]).columns
    summary_rows: list[dict[str, float | int | str]] = []
    bounds: dict[str, tuple[float, float]] = {}

    for column in numeric_columns:
        q1 = X[column].quantile(0.25)
        q3 = X[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_mask = (X[column] < lower_bound) | (X[column] > upper_bound)

        bounds[column] = (float(lower_bound), float(upper_bound))
        summary_rows.append(
            {
                "feature": column,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "outlier_count": int(outlier_mask.sum()),
                "outlier_ratio": float(outlier_mask.mean()),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(by=["outlier_count", "outlier_ratio"], ascending=False)
    return summary, bounds


def cap_outliers_iqr(X_train: pd.DataFrame, X_test: pd.DataFrame, bounds):
    X_train_capped = X_train.copy()
    X_test_capped = X_test.copy()

    for column, (lower_bound, upper_bound) in bounds.items():
        X_train_capped[column] = X_train_capped[column].clip(lower=lower_bound, upper=upper_bound)
        X_test_capped[column] = X_test_capped[column].clip(lower=lower_bound, upper=upper_bound)

    return X_train_capped, X_test_capped


# def save_dataset_overview(
#     df: pd.DataFrame,
#     feature_scores: pd.Series,
#     dropped_features: list[str],
#     manually_removed_features: list[str],
#     outlier_summary: pd.DataFrame,
# ) -> None:
#     overview_path = OUTPUT_DIR / "dataset_summary.txt"
#     with overview_path.open("w", encoding="utf-8") as file:
#         file.write(f"Rows: {df.shape[0]}\n")
#         file.write(f"Columns: {df.shape[1]}\n")
#         file.write(f"Missing values: {int(df.isna().sum().sum())}\n")
#         all_removed = manually_removed_features + dropped_features
#         file.write(f"Dropped redundant features: {', '.join(all_removed) or 'None'}\n\n")
#         file.write("Top outlier counts from training data (IQR method):\n")
#         file.write(
#             outlier_summary.head(10)[["feature", "outlier_count", "outlier_ratio"]]
#             .round(4)
#             .to_string(index=False)
#         )
#         file.write("\n\n")
#         file.write("Top mutual information feature scores:\n")
#         file.write(feature_scores.head(18).round(6).to_string())


def plot_target_distribution(y: pd.Series):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(y, bins=30, color="#2a9d8f", edgecolor="black", alpha=0.85)
    ax.set_title("Vitamin Deficiency Target Distribution")
    ax.set_xlabel("vitamin_deficiency")
    ax.set_ylabel("Frequency")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR + "\\target_distribution.png", dpi=200)
    plt.close(fig)


def plot_outlier_boxplots(X: pd.DataFrame, outlier_summary: pd.DataFrame, top_n: int = 6):
    top_features = outlier_summary.head(top_n)["feature"].tolist()
    if not top_features:
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()

    for index, column in enumerate(top_features):
        axes[index].boxplot(
            X[column],
            vert=True,
            patch_artist=True,
            boxprops={"facecolor": "#a8dadc"},
        )
        axes[index].set_title(column)
        axes[index].set_ylabel("Value")
        axes[index].grid(alpha=0.2)

    for index in range(len(top_features), len(axes)):
        axes[index].axis("off")

    fig.suptitle("Top Numeric Features by Outlier Count", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR + "\\outlier_boxplots.png", dpi=220)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, selected_columns: list[str]) -> None:
    correlation_frame = df[selected_columns + [TARGET_COLUMN]].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(14, 10))
    image = ax.imshow(correlation_frame.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(correlation_frame.columns)))
    ax.set_yticks(range(len(correlation_frame.columns)))
    ax.set_xticklabels(correlation_frame.columns, rotation=90)
    ax.set_yticklabels(correlation_frame.columns)
    ax.set_title("Correlation Heatmap for Selected Features")

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Correlation")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR + "\\correlation_heatmap.png", dpi=220)
    plt.close(fig)


def build_models() -> dict[str, object]:
    elastic_net = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=0.01, l1_ratio=0.35, max_iter=10000)),
        ]
    )

    models = {
        "ElasticNet": TransformedTargetRegressor(
            regressor=elastic_net, func=np.log1p, inverse_func=np.expm1
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=400,
            max_depth=10,
            min_samples_split=6,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }
    return models


def evaluate_models(models, X_train, X_test, y_train, y_test):
    rows = []
    predictions = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = np.clip(model.predict(X_test), a_min=0, a_max=None)
        predictions[model_name] = y_pred

        rows.append(
            {
                "model": model_name,
                "r2_score": r2_score(y_test, y_pred),
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "mae": mean_absolute_error(y_test, y_pred),
            }
        )

    metrics = pd.DataFrame(rows).sort_values(by="r2_score", ascending=False).reset_index(
        drop=True
    )
    return metrics, predictions


def plot_model_predictions(y_test, predictions, metrics):
    y_test_array = y_test.to_numpy()
    sorted_order = np.argsort(y_test_array)

    fig, axes = plt.subplots(3, 2, figsize=(15, 16))
    axes = axes.ravel()

    for index, model_name in enumerate(predictions):
        y_pred = predictions[model_name]
        model_r2 = metrics.loc[metrics["model"] == model_name, "r2_score"].iloc[0]

        scatter_ax = axes[index * 2]
        scatter_ax.scatter(y_test_array, y_pred, alpha=0.7, color="#264653")
        min_value = min(y_test_array.min(), y_pred.min())
        max_value = max(y_test_array.max(), y_pred.max())
        scatter_ax.plot([min_value, max_value], [min_value, max_value], "--", color="#e76f51")
        scatter_ax.set_title(f"{model_name}: Actual vs Predicted")
        scatter_ax.set_xlabel("Actual")
        scatter_ax.set_ylabel("Predicted")
        scatter_ax.text(
            0.03,
            0.94,
            f"R2 = {model_r2:.3f}",
            transform=scatter_ax.transAxes,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

        line_ax = axes[index * 2 + 1]
        line_ax.plot(y_test_array[sorted_order], label="Actual", color="#2a9d8f", linewidth=2)
        line_ax.plot(y_pred[sorted_order], label="Predicted", color="#e76f51", alpha=0.85)
        line_ax.set_title(f"{model_name}: Sorted Test vs Prediction")
        line_ax.set_xlabel("Sorted Test Samples")
        line_ax.set_ylabel(TARGET_COLUMN)
        line_ax.legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "model_predictions.png", dpi=220)
    plt.close(fig)


# def save_results(
#     X_train: pd.DataFrame,
#     X_test: pd.DataFrame,
#     y_test: pd.Series,
#     metrics: pd.DataFrame,
#     predictions: dict[str, np.ndarray],
#     feature_scores: pd.Series,
#     outlier_summary: pd.DataFrame):
#     metrics.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
#     feature_scores.rename("mutual_information").to_csv(OUTPUT_DIR / "feature_scores.csv")
#     outlier_summary.to_csv(OUTPUT_DIR / "outlier_summary.csv", index=False)

#     prediction_frame = pd.DataFrame({"actual": y_test.reset_index(drop=True)})
#     for model_name, y_pred in predictions.items():
#         prediction_frame[f"{model_name}_predicted"] = y_pred
#     prediction_frame.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

#     selected_features = pd.DataFrame({"selected_feature": X_train.columns})
#     selected_features.to_csv(OUTPUT_DIR / "selected_features.csv", index=False)

#     train_test_shapes = pd.DataFrame(
#         {
#             "split": ["train", "test"],
#             "rows": [X_train.shape[0], X_test.shape[0]],
#             "columns": [X_train.shape[1], X_test.shape[1]],
#         }
#     )
#     train_test_shapes.to_csv(OUTPUT_DIR / "train_test_shapes.csv", index=False)


def main():
    df = pd.read_csv(DATA_PATH)

    plot_target_distribution(df[TARGET_COLUMN])

    engineered_df = engineer_features(df)

    X = engineered_df.drop(columns=[TARGET_COLUMN])
    y = engineered_df[TARGET_COLUMN]

    X_clean, dropped_features = remove_unwanted_features(X)

    X_train, X_test, y_train, y_test = train_test_split(X_clean, y, test_size=0.2, random_state=RANDOM_STATE)

    outlier_summary, outlier_bounds = detect_outliers_iqr(X_train)
    plot_outlier_boxplots(X_train, outlier_summary)
    X_train_capped, X_test_capped = cap_outliers_iqr(X_train, X_test, outlier_bounds)

    X_train_selected, X_test_selected, feature_scores = select_features(X_train_capped, y_train, X_test_capped)
    plot_correlation_heatmap(pd.concat([X_train_capped[X_train_selected.columns], y_train], axis=1), X_train_selected.columns.tolist())

    models = build_models()

    metrics, predictions = evaluate_models(models, X_train_selected, X_test_selected, y_train, y_test)
    plot_model_predictions(y_test, predictions, metrics)

    # save_results(X_train_selected, X_test_selected, y_test, metrics, predictions, feature_scores, outlier_summary)

    print("Regression pipeline complete.")
    print(f"Outputs saved to: {OUTPUT_DIR.resolve()}")
    print("\nModel performance:")
    print(metrics.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
