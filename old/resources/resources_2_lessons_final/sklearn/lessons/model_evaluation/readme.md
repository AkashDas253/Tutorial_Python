# Model Evaluation

## Metrics for Classification
Metrics for classification are used to evaluate the performance of classification models.

#### Syntax

```python
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
report = classification_report(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='binary')
precision = precision_score(y_true, y_pred, average='binary')
recall = recall_score(y_true, y_pred, average='binary')
roc_auc = roc_auc_score(y_true, y_score)
```

#### Parameters
- **`y_true`**: Ground truth (correct) target values.
- **`y_pred`**: Estimated targets as returned by a classifier.
- **`average='binary'`**: {None, 'binary', 'micro', 'macro', 'samples', 'weighted'} Determines the type of averaging performed on the data.
- **`y_score`**: Target scores, can either be probability estimates of the positive class, confidence values, or binary decisions.

#### Attributes
- **`classification_report`**: Build a text report showing the main classification metrics.
- **`accuracy_score`**: Accuracy classification score.
- **`f1_score`**: Compute the F1 score, also known as balanced F-score or F-measure.
- **`precision_score`**: Compute the precision.
- **`recall_score`**: Compute the recall.
- **`roc_auc_score`**: Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.

#### Functions
- **`classification_report(y_true, y_pred, target_names=None, sample_weight=None, digits=2, output_dict=False, zero_division='warn')`**: Generate a classification report.
- **`accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)`**: Calculate the accuracy score.
- **`f1_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')`**: Calculate the F1 score.
- **`precision_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')`**: Calculate the precision score.
- **`recall_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')`**: Calculate the recall score.
- **`roc_auc_score(y_true, y_score, average='macro', sample_weight=None, max_fpr=None, multi_class='raise', labels=None)`**: Calculate the ROC AUC score.

## Metrics for Regression
Metrics for regression are used to evaluate the performance of regression models.

#### Syntax

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mse = mean_squared_error(y_true, y_pred, sample_weight=None, multioutput='uniform_average', squared=True)
mae = mean_absolute_error(y_true, y_pred, sample_weight=None, multioutput='uniform_average')
r2 = r2_score(y_true, y_pred, sample_weight=None, multioutput='uniform_average')
```

#### Parameters
- **`y_true`**: Ground truth (correct) target values.
- **`y_pred`**: Estimated target values.
- **`sample_weight=None`**: Sample weights. {None, array-like}
- **`multioutput='uniform_average'`**: Defines aggregating of multiple output values. {'raw_values', 'uniform_average', 'variance_weighted'}
- **`squared=True`**: If True returns MSE value, if False returns RMSE value. {True, False} (only for `mean_squared_error`)

#### Attributes
- **`mean_squared_error`**: Mean squared error regression loss.
- **`mean_absolute_error`**: Mean absolute error regression loss.
- **`r2_score`**: R^2 (coefficient of determination) regression score function.

#### Functions
- **`mean_squared_error(y_true, y_pred, sample_weight=None, multioutput='uniform_average', squared=True)`**: Compute mean squared error.
- **`mean_absolute_error(y_true, y_pred, sample_weight=None, multioutput='uniform_average')`**: Compute mean absolute error.
- **`r2_score(y_true, y_pred, sample_weight=None, multioutput='uniform_average')`**: Compute R^2 score.

### Confusion Matrix
A confusion matrix is a table used to evaluate the performance of a classification algorithm by comparing the actual and predicted classifications.

#### Syntax

```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred, labels=None, sample_weight=None, normalize=None)
```

#### Parameters
- **`y_true`**: Ground truth (correct) target values.
- **`y_pred`**: Estimated targets as returned by a classifier.
- **`labels=None`**: List of labels to index the matrix. {None, array-like}
- **`sample_weight=None`**: Sample weights. {None, array-like}
- **`normalize=None`**: Normalizes confusion matrix over the true (rows), predicted (columns) conditions or all the population. {None, 'true', 'pred', 'all'}

#### Attributes
- **`tn`**: True negatives.
- **`fp`**: False positives.
- **`fn`**: False negatives.
- **`tp`**: True positives.

#### Functions
- **`confusion_matrix(y_true, y_pred, labels=None, sample_weight=None, normalize=None)`**: Compute confusion matrix to evaluate the accuracy of a classification.

### ROC Curve
ROC (Receiver Operating Characteristic) curve is a graphical plot that illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied.

#### Syntax

```python
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=None, sample_weight=None, drop_intermediate=True)
```

#### Parameters
- **`y_true`**: True binary labels in range {0, 1} or {-1, 1}.
- **`y_score`**: Target scores, can either be probability estimates of the positive class, confidence values, or binary decisions.
- **`pos_label=None`**: The label of the positive class. {None, int, str}
- **`sample_weight=None`**: Sample weights. {None, array-like}
- **`drop_intermediate=True`**: Whether to drop some suboptimal thresholds which would not appear on a plotted ROC curve. {True, False}

#### Attributes
- **`fpr`**: Array of false positive rates.
- **`tpr`**: Array of true positive rates.
- **`thresholds`**: Array of thresholds used to compute `fpr` and `tpr`.

#### Functions
- **`roc_curve(y_true, y_score, pos_label=None, sample_weight=None, drop_intermediate=True)`**: Compute Receiver operating characteristic (ROC).
