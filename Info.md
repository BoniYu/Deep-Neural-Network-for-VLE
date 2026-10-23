# Info

## Critical point

The critical point is the end of the vapor-liquid coexistence curve, defined by the critical temperature $T_c$ and critical pressure $P_c$.

- The liquid and vapor densities become equal, so the two phases are indistinguishable.
- The enthalpy of vaporization goes to zero: $\Delta H_{vap} \to 0$ as $T \to T_c$.
- Above $T_c$ there is a single supercritical fluid; no pressure can condense it into a separate liquid.

**In the CO2 data:** the row where `H_vap = 0` is the critical point (about 31.1 °C and 7391 kPa).

## Clapeyron and Clausius-Clapeyron equations

Along the vapor-liquid coexistence curve, the exact Clapeyron equation relates the slope of the curve to the enthalpy and volume change of vaporization:

$$
\frac{dP}{dT} = \frac{\Delta H_{vap}}{T \, \Delta V}
$$

Clausius-Clapeyron is the approximation obtained by assuming:

1. the liquid volume is negligible, so $\Delta V \approx V_{vap}$
2. the vapor is an ideal gas, so $V_{vap} = RT/P$
3. $\Delta H_{vap}$ is constant

This gives

$$
\ln P = -\frac{\Delta H_{vap}}{R T} + c
$$

so $\ln P$ against $1/T$ is a straight line with slope $-\Delta H_{vap}/R$.

**Limitation:** all three assumptions fail near the critical point, where $\Delta H_{vap} \to 0$.

# Second part Neural Network

## Molecular fingerprint

A molecular fingerprint is a vector of 0s and 1s that describes a molecule's structure. It is computed from the SMILES string, and each bit marks whether a particular substructure (chain, ring, functional group, and so on) is present (1) or absent (0). In this project each fingerprint has 2048 bits.

- **Depends only on structure, not on temperature.** A compound has the same fingerprint at every temperature. Only the temperature and vapor pressure change between its rows in the dataset.
- **Model input.** The input to the neural network is the fingerprint (2048 bits) plus the scaled temperature as one extra column, giving 2049 features.

### Why the fingerprint is needed

Each compound has its own vapor pressure curve $P_{vap}(T)$. Temperature says *where on the curve* a point is, and the fingerprint says *which curve*. With temperature alone, methane, ethanol and decane would all look identical to the model.

A name or ID would say which compound it is, but not what it is like. The fingerprint describes the structure, so the network can learn patterns such as:

- longer carbon chains and larger molecules tend to have lower vapor pressure (stronger dispersion forces)
- hydrogen-bonding groups such as $-\text{OH}$ lower the vapor pressure
- structurally related molecules have similar fingerprints and similar behavior

Because a new molecule is described by the same kinds of substructures, the trained network can estimate its vapor pressure without any measurements on it.

### Comparison with the Antoine equation

The Antoine equation has constants $A$, $B$, $C$ fitted separately for each compound, so it cannot be used on a compound without data. The network uses one set of weights for all compounds and gets compound-specific behavior from the fingerprint.

### Consequences

- **Split by compound.** All rows of one compound share a fingerprint, so a random row split would put the same molecule in both train and test. The tutorial splits by compound so the test set measures performance on unseen molecules.
- **Similarity.** The similarity (Tanimoto) between two molecules is one number per pair, because it doesn't depend on temperature. It can be used to tell how close a new molecule is to the training compounds.

**Limitation:** the fingerprint only records presence or absence of substructures. _Add your own notes on what information about a molecule this loses._


# Models used in the DNN for VLE project

Linear regression is not covered here. All models predict $\ln P_{vap}$ from a 2049-column input: a 2048-bit fingerprint plus the scaled temperature.

---

## 1. Deep Neural Network (DNN)

A **feed-forward neural network** (multilayer perceptron) is a chain of layers. Each layer takes the previous layer's output $\mathbf{x}$, multiplies it by a weight matrix $W$, adds a bias vector $\mathbf{b}$, and applies a non-linear **activation function** $\sigma$:

$$
\mathbf{h} = \sigma(W\mathbf{x} + \mathbf{b})
$$

The activation used here is **ReLU**, $\text{ReLU}(z) = \max(0, z)$. Without a non-linearity, stacking layers would still give a linear model. The final layer has no activation, so the output can be any real number (needed because $\ln P$ takes negative and positive values). A network with more than one hidden layer is called *deep*.

The weights and biases are the **trainable parameters**. They are adjusted to minimize a loss (here the mean squared error) using **backpropagation**, which computes the gradient of the loss with respect to every parameter, and an **optimizer** (Adam) that uses the gradients to update them.

**Strengths:** very flexible, can learn interactions between input features, and learns its own internal features.
**Weaknesses:** many choices to make (size, depth, learning rate, number of epochs), needs scaled inputs, overfits easily, and on tabular data like this it was beaten by gradient-boosted trees.

### 1.1 Model definition

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(DNN, self).__init__()
        self.layer_1 = nn.Linear(in_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x

model = DNN(X_train.shape[1], 50, 1)
```

| Part | Meaning |
|---|---|
| `nn.Module` | PyTorch's base class for models. By inheriting it, the model gets `.parameters()`, `.train()`, `.eval()`, `.state_dict()` and so on. |
| `__init__` | Runs once when the model is created. It **declares** the layers. |
| `super().__init__()` | Runs the parent class's setup. Without it, PyTorch cannot track the layers. |
| `in_dim` | Number of input features: 2049 (2048 fingerprint bits plus the scaled temperature). |
| `hidden_dim` | Number of neurons in each hidden layer. This was tuned (3, 10, 20, 50), and 50 was best. |
| `out_dim` | Number of outputs: 1 (the predicted $\ln P$). |
| `nn.Linear(a, b)` | A layer computing $W\mathbf{x} + \mathbf{b}$, with $a$ inputs and $b$ outputs. It has $a \times b$ weights and $b$ biases. |
| `self.layer_1`, `_2`, `_3` | The layers must have **distinct names**. Assigning two layers to the same name overwrites the first one (this gave 441 parameters instead of 41,441). |
| `forward(x)` | Defines how data flows through the layers (**use** the layers). It is called by `model(x)`. |
| `F.relu` | The ReLU activation, applied after the first two layers but **not** after the last. |

**Parameter count:** $(n_{in} + 1)h + (h + 1)h + (h + 1)$. For $n_{in} = 2049$ and $h = 50$: $102{,}500 + 2{,}550 + 51 = 105{,}101$. Check with:

```python
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

### 1.2 Preparing the data

```python
X_train_torch = torch.from_numpy(X_train).float()
y_train_torch = torch.from_numpy(y_train).float()
```

| Part | Meaning |
|---|---|
| `torch.from_numpy` | Converts a NumPy array to a PyTorch tensor. |
| `.float()` | Converts to 32-bit floats, which is the type of the network's weights. Mixing 64-bit data with a 32-bit model raises a dtype error. |
| `y` shape `(n, 1)` | The target must be a column to match the network's `(n, 1)` output. A flat `(n,)` array would be silently broadcast into an `(n, n)` grid and give a meaningless loss. |
| Device | Training runs on the CPU, which is PyTorch's default, so no `.to(device)` is needed. |

### 1.3 Loss function and optimizer

```python
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
```

| Part | Meaning |
|---|---|
| `nn.MSELoss()` | Mean squared error, $\frac{1}{n}\sum (y_i - \hat{y}_i)^2$. This is the quantity minimized during training. |
| `Adam` | An optimizer that keeps running averages of each parameter's gradient and squared gradient, so every parameter gets its own adaptive step size. It usually works well with little tuning. |
| `model.parameters()` | Tells the optimizer **which** parameters to update. It must come from the **same model** that is being trained, otherwise the weights never change and the loss stays flat. |
| `lr=0.002` | The **learning rate**: the size of each update step. Too large and the loss jumps around, too small and training is slow. |

### 1.4 Training loop

```python
n_epochs = 250
losses = np.zeros((n_epochs, 2))

for epoch in range(n_epochs):
    model.train()
    # forward propagation
    y_train_pred = model(X_train_torch)
    loss = loss_function(y_train_pred, y_train_torch)
    losses[epoch, 0] = loss.item()

    # backward propagation
    model.zero_grad()
    loss.backward()

    # update the parameters
    optimizer.step()

    # validation loss
    model.eval()
    with torch.no_grad():
        y_valid_pred = model(X_valid_torch)
        loss_valid = loss_function(y_valid_pred, y_valid_torch)
        losses[epoch, 1] = loss_valid.item()
```

| Part | Meaning |
|---|---|
| `n_epochs` | One epoch is one pass over the training data. Here each epoch is a **single full-batch update** (all 26,310 rows at once). |
| `model.train()` / `model.eval()` | Switch the model's mode. They matter for layers such as dropout and batch normalization. This network has none, but the habit is kept. |
| Forward propagation | `model(X)` computes the predictions, and the loss compares them with the true values. |
| `loss.item()` | Converts the one-value loss tensor to a plain Python number, for storing and printing. |
| `model.zero_grad()` | Clears the gradients from the previous step. PyTorch **accumulates** gradients by default, so without this, each step would add to the old ones. |
| `loss.backward()` | Backpropagation: computes the gradient of the loss with respect to every parameter. It frees the computation graph afterwards, so each epoch needs its own forward pass (calling `backward` on an old, reused graph raises the "backward through the graph a second time" error). |
| `optimizer.step()` | Updates the parameters using the gradients. |
| `torch.no_grad()` | Turns off gradient tracking for the validation step, which saves memory and time, since no update happens there. |

**Overfitting:** the training loss keeps falling, while the validation loss reaches a minimum and then rises. The fix is early stopping.

### 1.5 Early stopping

```python
import copy

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state = None

    def step(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
        return self.counter >= self.patience

stopper = EarlyStopping(patience=30)
# inside the epoch loop, after the validation loss:
#     if stopper.step(loss_valid.item(), model):
#         break
# after the loop:
model.load_state_dict(stopper.best_state)
```

| Part | Meaning |
|---|---|
| `patience` | How many epochs to wait without a **new best** validation loss before stopping (30 here). It exists because the validation curve is noisy. |
| `min_delta` | The smallest improvement that counts as a new best (0 means any improvement). |
| `best_loss` | The lowest validation loss seen so far, starting at infinity. |
| `counter` | Epochs since the last new best. |
| `state_dict()` | The model's weights, as a dictionary. |
| `copy.deepcopy` | Essential: without it you keep a reference that changes as training continues, so the "best" snapshot would silently become the latest weights. |
| `load_state_dict` | Restores the best-epoch weights after the loop. The model at the stopping epoch is already past its best point. |
| `.item()` | Passes a plain number, so the stopper does not hold on to tensors and their graphs. |

The stopper must be created **fresh for every run** (and for every hidden size in the tuning loop), together with a fresh model and optimizer.

### 1.6 Saving and loading

```python
torch.save(model.state_dict(), "models/dnn_vle.pt")
joblib.dump(scaler, "models/temperature_scaler.joblib")

loaded = DNN(in_dim, hidden_dim, out_dim)
loaded.load_state_dict(torch.load("models/dnn_vle.pt"))
loaded.eval()
```

Only the weights (`state_dict`) are saved, so the class must be defined (here in `src/model.py`) with the same layer names and sizes before loading. The fitted temperature scaler is saved too, since any prediction needs the same scaling as in training. `config.json` stores the sizes and settings.

### 1.7 Hyperparameter summary

| Hyperparameter | Value used | Notes |
|---|---|---|
| Hidden layers | 2 | fixed by the class |
| `hidden_dim` | 50 | tuned over 3, 10, 20, 50 |
| Activation | ReLU | not on the output layer |
| Loss | MSE | |
| Optimizer / `lr` | Adam / 0.002 | |
| Epochs | up to 250 | stopped early by the stopper |
| Batch | full training set | one update per epoch |
| `patience` | 30 | |
| Seed | `torch.manual_seed(0)` | makes the initial weights reproducible |

---

## 2. XGBoost

**XGBoost** (extreme gradient boosting) builds an ensemble of decision trees **one after another**. Each new tree is fitted to correct the errors the current ensemble still makes. The prediction is the sum of all the trees' outputs, with each tree's contribution scaled by the learning rate $\eta$:

$$
\hat{y}_i = \sum_{k=1}^{K} \eta \, f_k(\mathbf{x}_i)
$$

Training minimizes a **regularized objective**: the loss plus a penalty on the complexity of the trees:

$$
\text{Obj} = \sum_i \ell(y_i, \hat{y}_i) + \sum_k \Omega(f_k), \qquad \Omega(f) = \gamma T + \tfrac{1}{2}\lambda \sum_j w_j^2 + \alpha \sum_j |w_j|
$$

where $T$ is the number of leaves and $w_j$ are the leaf values. For the squared-error loss used here, each new tree is fitted to the current **residuals** (true minus predicted values), so the ensemble improves a little at every step.

**Strengths:** very strong on tabular data, handles non-linear effects and interactions, needs no feature scaling, fast, and has built-in regularization.
**Weaknesses:** many hyperparameters, and, as with any tree model, it cannot predict outside the range of values it has seen in training (the predictions are built from leaf values).

**Result in this project:** the best model. Tuned XGBoost test MAE 1.230 and $R^2$ 0.781, against 1.634 and 0.642 for the DNN.

### 2.1 Default model

```python
from xgboost import XGBRegressor

xgb = XGBRegressor(random_state=42, n_jobs=-1)
xgb.fit(X_train, y_train.ravel())
y_valid_pred = xgb.predict(X_valid)
```

| Part | Meaning |
|---|---|
| `XGBRegressor` | The regression version. The classification version is `XGBClassifier`. |
| `random_state=42` | Fixes the randomness (used by row and column subsampling), so results can be reproduced. |
| `n_jobs=-1` | Use all CPU cores. |
| `fit(X, y.ravel())` | `.ravel()` turns the `(n, 1)` target into a flat `(n,)` array, which scikit-learn-style models expect. |
| No scaling | Trees only compare a feature with a threshold, so the scale of the inputs doesn't matter. |

The main parameters, with their default values:

| Parameter | Default | What it controls |
|---|---|---|
| `n_estimators` | 100 | Number of trees (boosting rounds). More trees fit the data more closely, and can overfit. |
| `learning_rate` ($\eta$) | 0.3 | Shrinks each tree's contribution. Smaller values need more trees but generalize better. The two work as a pair: total fitting is roughly `learning_rate × n_estimators`. |
| `max_depth` | 6 | Maximum depth of each tree. Deeper trees capture more complex interactions and overfit more easily. |
| `min_child_weight` | 1 | Minimum total weight in a leaf. Larger values stop the trees from creating tiny leaves, so the model becomes more conservative. |
| `subsample` | 1.0 | Fraction of training **rows** used for each tree. Less than 1 adds randomness and reduces overfitting. |
| `colsample_bytree` | 1.0 | Fraction of **features** (the 2049 columns) used for each tree. |
| `reg_lambda` ($\lambda$) | 1 | L2 penalty on the leaf values. Larger values make the model smoother. |
| `reg_alpha` ($\alpha$) | 0 | L1 penalty on the leaf values. Can push leaf values to zero. |

### 2.2 Hyperparameter tuning with GroupKFold

```python
from scipy.stats import randint, uniform, loguniform
from sklearn.model_selection import GroupKFold, RandomizedSearchCV

groups_train = df_train['Name'].to_numpy()

param_distributions = {
    "n_estimators":     randint(200, 1001),
    "learning_rate":    loguniform(0.02, 0.3),
    "max_depth":        randint(3, 9),
    "min_child_weight": randint(1, 11),
    "subsample":        uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.3, 0.7),
    "reg_lambda":       loguniform(0.1, 10),
}

search = RandomizedSearchCV(
    estimator=XGBRegressor(random_state=42, n_jobs=-1),
    param_distributions=param_distributions,
    n_iter=30,
    scoring={"MAE": "neg_mean_absolute_error",
             "RMSE": "neg_root_mean_squared_error"},
    refit="MAE",
    cv=GroupKFold(n_splits=5),
    random_state=42,
    verbose=2,
    return_train_score=True,
)
search.fit(X_train, y_train.ravel(), groups=groups_train)
```

| Part | Meaning |
|---|---|
| `RandomizedSearchCV` | Tries randomly **sampled** parameter combinations. It covers many parameters better than a full grid, whose size multiplies with every parameter added. |
| `param_distributions` | For each parameter, the distribution to sample from. |
| `randint(a, b)` | A random integer from $a$ to $b-1$. |
| `uniform(loc, scale)` | A random number from `loc` to `loc + scale`, so `uniform(0.6, 0.4)` is 0.6 to 1.0. |
| `loguniform(a, b)` | A random number from $a$ to $b$ with equal chance on a log scale. It suits parameters that range over orders of magnitude (the learning rate, $\lambda$). |
| `n_iter=30` | Number of combinations tried. With 5 folds this means 150 model fits. |
| `scoring` | Metrics computed. scikit-learn treats **higher as better**, so errors are negated (`neg_`), and the printed scores are negative MAE and RMSE. |
| `refit="MAE"` | Which metric picks the winner. The best model is then refitted on the full training set (`search.best_estimator_`). |
| `cv=GroupKFold(5)` | 5-fold cross-validation in which **a compound never appears in two folds**. |
| `groups=` | The group label per row (the compound name). It must line up with the rows of `X_train`. Ordinary k-fold would put rows of the same compound (same fingerprint) on both sides and reward memorizing molecules. |
| `return_train_score` | Also stores the training-fold scores, to see overfitting. |
| `verbose=2` | Prints each fit as it runs, a way to see progress. |

Results are read from `search.best_params_`, `search.best_score_` (negative MAE) and `pd.DataFrame(search.cv_results_)`.

### 2.3 Best parameters found

| Parameter | Value | Default |
|---|---|---|
| `n_estimators` | 400 | 100 |
| `learning_rate` | 0.117 | 0.3 |
| `max_depth` | 6 | 6 |
| `min_child_weight` | 7 | 1 |
| `subsample` | 0.732 | 1.0 |
| `colsample_bytree` | 0.948 | 1.0 |
| `reg_lambda` | 1.76 | 1 |

Cross-validated MAE: 1.298. The model uses more trees with a lower learning rate, a larger minimum leaf weight, and rows sampled at about 73% per tree, which makes it more conservative than the default.

### 2.4 Saving and loading

```python
best_xgb = search.best_estimator_
best_xgb.save_model("models/xgb_tuned.json")

loaded_xgb = XGBRegressor()
loaded_xgb.load_model("models/xgb_tuned.json")
```

`save_model` writes XGBoost's own JSON format, which works across XGBoost versions better than a pickle. The tuned parameters are stored separately in `xgb_config.json` (NumPy numbers have to be converted with `int()` or `float()` before they can go into JSON).

---

## 3. Random Forest

A **random forest** is an ensemble of many deep decision trees trained **independently**. Each tree sees a **bootstrap sample** of the rows (random sampling with replacement), and at every split only a random subset of the features is considered. The forest's prediction is the **average** of all the trees' predictions. A single deep tree overfits, but averaging many different trees cancels much of the variance, and this is called **bagging**.

**Strengths:** robust, few settings that need tuning, hard to overfit badly, and handles non-linear effects.
**Weaknesses:** slow to fit with many features and rows (about 190 to 250 s here, against a few seconds for XGBoost), large models, and it cannot extrapolate beyond the training values.

### 3.1 Code

```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state=42, n_jobs=-1)
rf.fit(X_train, y_train.ravel())
```

| Parameter | Default | Meaning |
|---|---|---|
| `n_estimators` | 100 | Number of trees. More trees reduce the variance, with diminishing returns, and increase the time. |
| `max_depth` | None | Trees grow until their leaves are pure (or too small to split). Limiting the depth reduces overfitting. |
| `min_samples_leaf` | 1 | Minimum rows in a leaf. Larger values give smoother trees. |
| `max_features` | 1.0 (all) | Fraction of features considered at each split. Lower values make the trees more different from each other. |
| `bootstrap` | True | Whether each tree uses a bootstrap sample of the rows. |
| `n_jobs=-1` | | Fit trees in parallel on all cores. |
| `random_state=42` | | Reproducible sampling. |

---

## 4. Decision Tree

A **decision tree** splits the data again and again with yes/no questions on single features ("is bit 512 set?"). At each node it chooses the feature and threshold that most reduce the squared error (the variance of $\ln P$) in the two resulting groups. A **leaf** predicts the **mean** of the training targets that ended up in it.

**Strengths:** simple, fast, needs no scaling, and easy to interpret.
**Weaknesses:** with default settings the tree grows until it fits the training data almost exactly (training MAE 0.002, validation MAE 1.745), so it overfits completely. It is the building block of both the forest and XGBoost.

### 4.1 Code

```python
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(random_state=42)
tree.fit(X_train, y_train.ravel())
```

| Parameter | Default | Meaning |
|---|---|---|
| `max_depth` | None | Maximum depth. Limiting it is the simplest way to control overfitting. |
| `min_samples_split` | 2 | Minimum rows needed to split a node. |
| `min_samples_leaf` | 1 | Minimum rows in a leaf. |
| `criterion` | squared error | The measure of how good a split is. |
| `random_state=42` | | Fixes how ties between equally good splits are broken, so the tree is reproducible. |

---

## 5. Comparison

| | Decision Tree | Random Forest | XGBoost | DNN |
|---|---|---|---|---|
| How it works | one tree | many deep trees, **averaged** | many shallow trees built **one after another**, each correcting the last | layers of weights and ReLU activations |
| Needs scaled inputs | no | no | no | yes |
| Main settings | `max_depth`, `min_samples_leaf` | `n_estimators`, `max_features`, `max_depth` | `n_estimators`, `learning_rate`, `max_depth`, `subsample`, regularization | `hidden_dim`, layers, `lr`, epochs, `patience` |
| Overfitting control | limit the depth | averaging | learning rate, regularization, sampling | early stopping |
| Fit time (here) | 8 to 41 s | 190 to 250 s | 4 to 6 s | not measured |
| Test MAE | 1.650 | 1.353 | 1.230 (tuned) | 1.634 |
| Test $R^2$ | 0.586 | 0.710 | 0.781 (tuned) | 0.642 |