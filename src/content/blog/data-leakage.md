---
title: "Your Model Scored 98%. Should You Trust It?"
description: "How data leakage, flawed model selection, and inappropriate evaluation splits can produce misleading machine learning results—and how to avoid them."
pubDate: 2026-10-23
tags:
  - Machine Learning
  - Data Science
  - Model Evaluation
  - Data Leakage
draft: false
---

A machine learning model achieves 98% accuracy. Its training pipeline looks reasonable, its validation curves are reassuring, and its results appear substantially better than those of competing approaches.

But there is a question we should ask before celebrating:

**Did the model learn a relationship that generalizes, or did our evaluation procedure give it information it should not have had?**

This is not merely a theoretical concern. Data leakage can produce overly optimistic performance estimates, undermine scientific conclusions, and make a model appear useful when its performance will not survive deployment. A cross-disciplinary review by [Kapoor and Narayanan (2023)](https://doi.org/10.1016/j.patter.2023.100804) identified leakage-related problems in at least 294 papers across 17 scientific fields. The authors also demonstrated how correcting leakage in a civil-war prediction study changed conclusions about the relative performance of complex machine learning models and logistic regression.

The lesson is not that high model scores are inherently suspicious. It is that a score is only meaningful when the procedure used to obtain it supports the claim we want to make.

In this article, we will examine how data leakage occurs, demonstrate its effects with a reproducible experiment, and develop a practical framework for trustworthy model evaluation.

## 1. What does a test score actually measure?

In supervised learning, we observe a dataset of input-output pairs,

$$
D = \{(x_i,y_i)\}_{i=1}^{n},
$$

where $x_i$ represents the features and $y_i$ the target. A learning algorithm uses the training data to produce a predictive model $f$.

Our goal is generally not to memorize the observed dataset. We want the model to make useful predictions on future observations drawn from the population in which it will be used.

Let $P$ denote the relevant data-generating distribution, and let $\ell$ be a loss function. The population risk of a model is

$$
R_P(f)
=
\mathbb{E}_{(X,Y)\sim P}
\left[\ell(f(X),Y)\right].
$$

For classification, we might use zero-one loss:

$$
\ell(f(X),Y)
=
\mathbb{1}\{f(X)\ne Y\},
$$

where $\mathbb{1}\{\cdot\}$ equals one when the stated condition holds and zero otherwise. In that setting, minimizing risk corresponds to minimizing the probability of misclassification.

We cannot usually calculate population risk directly because the true distribution $P$ is unknown. Instead, we estimate performance using held-out observations.

For a test set $D_{\text{test}}$ containing $m$ observations, empirical test risk is

$$
\widehat{R}_{\text{test}}(f)
=
\frac{1}{m}
\sum_{i=1}^{m}
\ell\left(f(x_i),y_i\right).
$$

This estimate is informative only under appropriate conditions. In particular, the test observations must represent the prediction setting of interest, and the model-development process must not exploit their evaluation labels.

A held-out test set is not automatically trustworthy simply because it is stored in a separate file. Feature selection, preprocessing, repeated model comparisons, and other decisions can still allow test information to influence the final result.

This brings us to the central issue: **evaluation is a property of the entire learning procedure, not just the final model.**

## 2. Data leakage: when information crosses the wrong boundary

Data leakage occurs when information that should be unavailable to a learning or evaluation step influences that step. The precise definition depends on the task and experimental design, but a common consequence is an overly optimistic estimate of generalization performance.

Leakage is not limited to accidentally including the target column among the features. It can occur during data preparation, feature engineering, sample construction, or model selection.

### 2.1. Preprocessing before the train-test split

Suppose we want to standardize numerical features, impute missing values, select informative features, or reduce dimensionality with principal component analysis (PCA).

These operations may estimate quantities from the observed data. For example, standardization typically estimates a feature's mean and standard deviation:

$$
z_j = \frac{x_j-\mu_j}{\sigma_j}.
$$

If $\mu_j$ and $\sigma_j$ are estimated from the complete dataset before splitting, the test features influence the transformation used to train the model.

This is a form of information leakage from the evaluation set into model development. It does not necessarily reveal the test labels, and its practical effect depends on the transformation and task, but it violates the intended separation.

Feature selection can create a more dramatic problem because it may directly use target labels.

The incorrect workflow is:

1. Fit a feature selector on the entire dataset.
2. Split the selected dataset into training and test sets.
3. Train a classifier and report its test score.

The correct workflow is:

1. Split the dataset.
2. Fit the feature selector using training data only.
3. Train the classifier on the transformed training data.
4. Apply the fitted transformation to the test data.
5. Evaluate the classifier on the untouched test labels.

The same principle applies to imputation, scaling, PCA, text vectorization, and other learned preprocessing operations. Scikit-learn documents these pitfalls and recommends pipelines to keep transformations within the training procedure.

### 2.2. Related observations appearing in both sets

Another failure occurs when training and test sets contain observations that are not sufficiently independent for the intended evaluation.

Imagine building an image classifier. Before splitting the dataset, we generate several augmented versions of each original image. Slightly rotated, cropped, or brightness-adjusted versions of the same source image may then appear in both training and test sets.

The test images are technically different files, but they share substantial information with the training images. The model may perform well because it has already encountered closely related examples.

Similar problems arise when:

- Multiple records from the same patient appear in both sets.
- The same customer contributes many correlated transactions to both sets.
- Near-duplicate documents are distributed across training and test data.
- Multiple recordings from the same speaker appear on both sides of a speech-recognition split.
- Measurements from the same experimental subject are treated as independent observations.

The appropriate response depends on what we want to predict. If the goal is generalization to entirely new patients, speakers, customers, or source images, the split should respect those groups.

For augmentation-based tasks, the usual safe sequence is to split the original observations first and then augment the training partition. Test data should remain representative of the evaluation task and should not contain transformed copies of training examples.

[Gerz and Jelali (2026)](https://doi.org/10.1016/j.atech.2026.102462) investigate the effects of augmentation sequencing in agricultural computer vision. Their study illustrates how augmenting before splitting can produce highly optimistic results when related images cross the evaluation boundary. The exact magnitude depends on the dataset and experimental setup; it should not be treated as a universal estimate of leakage effects.

### 2.3. Target leakage and information unavailable at prediction time

Sometimes the problem is not the split itself but the features we provide to the model.

Consider a model intended to predict whether a customer will default on a loan. If the feature table contains a value recorded only after default—such as a post-default collection outcome—the model can exploit information that would not exist when the prediction is needed.

This is often called target leakage or a form of temporal leakage.

The model may achieve excellent test performance if that feature is present in both training and test data. Yet the evaluation does not answer the real business question because the feature is unavailable at the intended prediction time.

For every feature, we should therefore ask:

- When was this value generated?
- Would it be available when the model makes a prediction?
- Could it encode the outcome or a consequence of the outcome?
- Was it constructed using information from outside the permitted training data?

A column can be statistically predictive and still be invalid for the intended task.

## 3. A reproducible experiment: obtaining 76% accuracy from random labels

Conceptual explanations are useful, but we can see the problem directly with a small experiment.

We generated a synthetic dataset containing 200 observations and 10,000 random features. Each observation received an independently generated binary label. Because the features and labels were generated independently, the data contained no genuine predictive relationship.

We then compared two procedures:

**Flawed procedure:** Select the 25 most promising features using the entire dataset, including the future test labels, before fitting a classifier.

**Correct procedure:** Split the data first and place feature selection inside a scikit-learn pipeline, so the selector learns only from the training data.

Both approaches used the same train-test split within each trial. We repeated the experiment across 20 independently generated datasets, using fixed seeds for reproducibility.

### The results

| Metric | Flawed evaluation | Correct pipeline |
|---|---:|---:|
| Mean test accuracy | 76.0% | 52.2% |
| Standard deviation | 5.1 pct points | 5.4 pct points |
| Minimum observed accuracy | 68.0% | 44.0% |
| Maximum observed accuracy | 84.0% | 60.0% |

The difference in mean accuracy was

$$
76.0\%-52.2\%=23.8
$$

percentage points **(pct points)**.

The flawed procedure averaged 76% accuracy even though the features contained no genuine predictive signal. The correct pipeline averaged 52.2%, much closer to the 50% baseline expected for balanced binary classification.

Why does this happen?

Feature selection examines how each feature relates to the target. When we perform selection before splitting, we give the selector access to the test labels. Among 10,000 random features, some will appear strongly associated with the labels purely by chance. Selecting the best-looking features across the complete dataset also selects features that happen to fit chance patterns in the test set.

The final classifier is therefore evaluated on observations whose labels have already influenced feature selection. The test set is no longer independent of the model-development procedure.

The pipeline prevents this particular failure by fitting the selector only on the training data.

Our correct-pipeline result is not exactly 50%, and it does not need to be. Each trial contains only 50 test observations, so observed accuracy naturally fluctuates around its expected value. The standard deviations reported here describe variation across the 20 trials; they are not confidence intervals.

This experiment demonstrates one specific form of evaluation leakage. It does not imply that every high-scoring model is contaminated, nor does it quantify the typical impact of leakage in real-world applications.

### The complete Python script

The following script reproduces the experiment using NumPy and scikit-learn. It prints the mean, standard deviation, range, and individual trial scores.

Save it as `data_leakage_demo.py`.

```python
import numpy as np

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

def run_trial(seed, n_samples=200, n_features=10_000, k=25):
    # Generate independent random features and binary labels.
    rng = np.random.RandomState(seed)
    X = rng.standard_normal((n_samples, n_features))
    y = rng.choice(2, n_samples)

    # Use the same split for both approaches.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed
    )

    # INCORRECT: feature selection sees all labels,
    # including those belonging to the future test set.
    selector = SelectKBest(k=k)
    X_selected = selector.fit_transform(X, y)

    X_train_leaky, X_test_leaky, y_train_leaky, y_test_leaky = (
        train_test_split(
            X_selected, y, test_size=0.25, random_state=seed
        )
    )

    leaky_model = HistGradientBoostingClassifier(random_state=1)
    leaky_model.fit(X_train_leaky, y_train_leaky)
    leaky_score = accuracy_score(y_test_leaky, leaky_model.predict(X_test_leaky))

    # CORRECT: the selector is fitted on training data only.
    safe_model = make_pipeline(
        SelectKBest(k=k),
        HistGradientBoostingClassifier(random_state=1),
    )

    safe_model.fit(X_train, y_train)
    safe_score = accuracy_score(y_test, safe_model.predict(X_test))

    return leaky_score, safe_score

def main():
    n_trials = 20

    results = np.array([run_trial(seed) for seed in range(n_trials)])

    leaky = results[:, 0]
    safe = results[:, 1]

    print(f"Trials: {n_trials}")

    print("\nMean test accuracy")
    print(f"  Leaky evaluation: {leaky.mean():.3f}")
    print(f"  Correct pipeline: {safe.mean():.3f}")

    print("\nStandard deviation across trials")
    print(f"  Leaky evaluation: {leaky.std(ddof=1):.3f}")
    print(f"  Correct pipeline: {safe.std(ddof=1):.3f}")

    print("\nRange across trials")
    print(
        f"  Leaky evaluation: "
        f"{leaky.min():.3f}–{leaky.max():.3f}"
    )
    print(
        f"  Correct pipeline: "
        f"{safe.min():.3f}–{safe.max():.3f}"
    )

    print("\nIndividual trials")
    for i, (bad, good) in enumerate(results, start=1):
        print(
            f"  Trial {i:02d}: "
            f"leaky={bad:.3f}, correct={good:.3f}"
        )

if __name__ == "__main__":
    main()
```

Install the dependencies if necessary:

```bash
pip install numpy scikit-learn
```

Then run:

```bash
python3 data_leakage_demo.py
```

The experiment reported in this article produced the following output:

```text
Trials: 20

Mean test accuracy
  Leaky evaluation: 0.760
  Correct pipeline: 0.522

Standard deviation across trials
  Leaky evaluation: 0.051
  Correct pipeline: 0.054

Range across trials
  Leaky evaluation: 0.680–0.840
  Correct pipeline: 0.440–0.600

Individual trials
  Trial 01: leaky=0.740, correct=0.480
  Trial 02: leaky=0.800, correct=0.480
  Trial 03: leaky=0.720, correct=0.480
  Trial 04: leaky=0.760, correct=0.440
  Trial 05: leaky=0.760, correct=0.520
  Trial 06: leaky=0.820, correct=0.580
  Trial 07: leaky=0.680, correct=0.480
  Trial 08: leaky=0.720, correct=0.560
  Trial 09: leaky=0.840, correct=0.440
  Trial 10: leaky=0.760, correct=0.580
  Trial 11: leaky=0.680, correct=0.440
  Trial 12: leaky=0.680, correct=0.560
  Trial 13: leaky=0.740, correct=0.480
  Trial 14: leaky=0.820, correct=0.580
  Trial 15: leaky=0.800, correct=0.520
  Trial 16: leaky=0.820, correct=0.540
  Trial 17: leaky=0.780, correct=0.600
  Trial 18: leaky=0.800, correct=0.560
  Trial 19: leaky=0.700, correct=0.520
  Trial 20: leaky=0.780, correct=0.600
```

The code uses fixed seeds, but exact results can depend on software versions and implementation details. The values above are the observed results of this run, not a claim that every environment will produce identical output.

Scikit-learn's own [common pitfalls documentation](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage-during-pre-processing) presents a closely related single-split example using random labels and feature selection. Our repeated experiment extends that illustration across multiple independently generated datasets.

## 4. Model-selection bias: the test set can be overused

Even if preprocessing is correct, the evaluation can become unreliable when we repeatedly use the same validation results to make development decisions.

Suppose we evaluate 100 candidate models on a validation set and select the one with the highest score. Each score contains some sampling noise. The best observed score may therefore reflect both genuine predictive quality and favorable noise in that particular validation sample.

The selected model is not necessarily the best model for the population. It is the model that performed best according to the criterion we optimized on the available validation data.

[Cawley and Talbot (2010)](https://www.jmlr.org/papers/v11/cawley10a.html) explain how overfitting a model-selection criterion can introduce selection bias in performance evaluation. [Varma and Simon (2006)](https://doi.org/10.1186/1471-2105-7-91) similarly show that evaluating a procedure on the same cross-validation results used to select its parameters can produce optimistically biased estimates.

This is related to data leakage, but the mechanism differs from fitting a feature selector on test labels. In model-selection bias, repeated adaptation to a finite validation sample makes the validation result less reliable as an independent estimate of generalization.

A sound workflow separates three roles:

- **Training data:** Fit model parameters and learned preprocessing.
- **Validation data:** Compare candidates, tune hyperparameters, and make development decisions.
- **Final test data:** Estimate performance after the development procedure has been finalized.

The final test set should not become another validation set. If we repeatedly inspect its results and change the model accordingly, we gradually turn it into part of the development process.

When data is limited, nested cross-validation can help estimate the performance of a complete model-selection procedure. An inner loop performs selection and tuning; an outer loop evaluates the selected procedure on held-out folds. It is not a magic guarantee, but it makes the separation between selection and evaluation explicit.

## 5. A correct split must match the deployment problem

A random train-test split is not always the right choice. The evaluation should reproduce the information boundaries and sampling conditions expected at deployment.

### Random split

A random split can be appropriate when observations are reasonably independent and deployment involves predictions for new observations drawn from the same population.

However, if the data contains repeated measurements from the same entity, a random split can distribute related observations across training and test sets. Group-based splitting may then be necessary.

### Group-based split

If we want to predict outcomes for entirely new patients, customers, speakers, or experimental subjects, observations from a single group should not appear in both training and test partitions.

The relevant group definition depends on the task. For example, splitting by patient may be appropriate for a new-patient prediction task, while splitting by hospital may be needed to investigate generalization to new hospitals.

### Temporal split

For forecasting or time-dependent prediction, randomly shuffling observations may allow the model to learn from the future and evaluate it on the past.

A more appropriate design trains on earlier observations and evaluates on later ones:

$$
D_{\text{train}} = \{(x_t,y_t):t\leq T\},
$$

$$
D_{\text{test}} = \{(x_t,y_t):t>T\}.
$$

The boundary $T$ should reflect the intended forecasting setup. If features depend on historical windows, those windows must also be constructed without using information from beyond the prediction time.

Temporal evaluation is not automatically leakage-free, but it makes the intended direction of prediction explicit.

### Leakage versus distribution shift

These concepts should not be conflated.

Leakage occurs when the learning or evaluation procedure uses information it should not have access to. Distribution shift occurs when the data distribution at deployment differs from the distribution represented during training or evaluation.

A model can have a perfectly separated test set and still fail under distribution shift. Conversely, leakage can produce an invalid estimate even when training and test observations come from the same underlying distribution.

A trustworthy evaluation should investigate both information leakage and whether the test data represents the conditions in which the model will be used.

## 6. A practical checklist for trustworthy evaluation

Before reporting a model's performance, work through the following questions.

### Data and features

- [ ] Is the target excluded from the input features unless it is legitimately available at prediction time?
- [ ] Are all features available at the intended prediction time?
- [ ] Could any feature encode a consequence of the target or an outcome recorded later?
- [ ] Have exact duplicates and near-duplicates been inspected where relevant?
- [ ] Are related observations grouped appropriately for the intended generalization task?

### Preprocessing and training

- [ ] Was the train-test split created before fitting learned preprocessing steps?
- [ ] Are imputation, scaling, feature selection, PCA, and vectorization fitted using training data only?
- [ ] Are learned transformations included in a pipeline where appropriate?
- [ ] Are synthetic or augmented samples generated only within the permitted training partition?
- [ ] Are cross-validation folds constructed to respect group or temporal constraints?

### Model selection and reporting

- [ ] Is validation data used for development rather than treated as an untouched final test?
- [ ] Is the final test set kept separate from repeated model selection?
- [ ] Does the evaluation split reflect the deployment scenario?
- [ ] Are uncertainty, variability, and limitations reported rather than presenting one score as definitive?
- [ ] Can the data preparation, split logic, model configuration, random seeds, and evaluation procedure be reproduced?

A checklist cannot detect every possible leakage mechanism. It does, however, turn evaluation integrity into an explicit part of the workflow rather than an assumption made after the score is computed.

## 7. One final distinction: evaluation leakage is not privacy leakage

The term *leakage* is also used in privacy and security research. For example, [Zhang, Tople, and Ohrimenko (2021)](https://www.usenix.org/conference/usenixsecurity21/presentation/zhang-wanrong) studied how outputs from multi-party machine learning could reveal properties of another party's dataset.

That problem is related to information exposure, but it is different from the train-test contamination discussed in this article. Evaluation leakage undermines the validity of performance estimates; privacy leakage concerns information that can be inferred or exposed through a model or learning protocol.

A model may have a trustworthy test score and still pose privacy risks. Similarly, a model can have no obvious privacy issue and still have an invalid evaluation.

These are separate questions that deserve separate analyses.

## Conclusion

A high test score is evidence, not a guarantee.

Its meaning depends on how the data was collected, how observations were separated, which transformations were fitted, how models were selected, and whether the evaluation resembles the intended prediction setting.

Our synthetic experiment makes one point particularly clear: a flawed evaluation procedure can produce 76% accuracy from random features and random labels, while a correctly constructed pipeline brings performance back toward chance. The apparent success came not from a useful predictive relationship, but from allowing test labels to influence feature selection.

The remedy is not simply to reserve a test set. We must preserve the test set's role throughout the entire development process.

**Trustworthy machine learning begins with trustworthy evaluation.** Before asking whether a model is accurate, we should ask whether our experiment gives us a valid reason to believe that it is.

---

## References

1. Kapoor, S., & Narayanan, A. (2023). [Leakage and the reproducibility crisis in machine-learning-based science](https://doi.org/10.1016/j.patter.2023.100804). *Patterns, 4*(9), 100804.

2. Apicella, A., Isgrò, F., & Prevete, R. (2025). [Don't push the button! Exploring data leakage risks in machine learning and transfer learning](https://doi.org/10.1007/s10462-025-11326-3). *Artificial Intelligence Review*.

3. Gerz, F., & Jelali, M. (2026). [Towards reliable data augmentation in machine learning: Practices to prevent data leakage](https://doi.org/10.1016/j.atech.2026.102462). *Smart Agricultural Technology*, 15, 102462.

4. Cawley, G. C., & Talbot, N. L. C. (2010). [On over-fitting in model selection and subsequent selection bias in performance evaluation](https://www.jmlr.org/papers/v11/cawley10a.html). *Journal of Machine Learning Research, 11*, 2079–2107.

5. Varma, S., & Simon, R. (2006). [Bias in error estimation when using cross-validation for model selection](https://doi.org/10.1186/1471-2105-7-91). *BMC Bioinformatics, 7*, 91.

6. Scikit-learn developers. [Common pitfalls and recommended practices: Data leakage during preprocessing](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage-during-pre-processing). Scikit-learn documentation.

7. Zhang, W., Tople, S., & Ohrimenko, O. (2021). [Leakage of Dataset Properties in Multi-Party Machine Learning](https://www.usenix.org/conference/usenixsecurity21/presentation/zhang-wanrong). *30th USENIX Security Symposium*.
