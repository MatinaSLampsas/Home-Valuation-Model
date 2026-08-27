# Predicting Home Prices: Hedonic Regression & Machine Learning
Built and optimized housing price prediction models on 12.5K observations using Random Forest, geospatial feature engineering, cross-validation, and interactive GIS mapping, achieving 75.7% R².
The project combines traditional econometric modeling, machine learning, and geospatial feature engineering across approximately **12,500 housing observations** to compare predictive performance and identify the factors most influential in home valuation.

---

## Project Overview

The objectives of this research were to:

- Compare traditional hedonic regression with tree-based machine learning models
- Identify the property and location characteristics most influential in housing prices
- Engineer geographic features using geocoded property data and GIS resources
- Evaluate predictive performance using out-of-sample testing and cross-validation
- Analyze residuals, feature importance, outliers, and potential overfitting
- Investigate systematic prediction errors for high-value properties
- Develop an interactive map comparing actual and predicted home prices

---

## Data

The initial dataset contained approximately **12,500 residential housing records** from San Luis Obispo County, California.

Property information was sourced from Multiple Listing Service (MLS) data and included internal and external housing characteristics.

### Property Features

Features used throughout the analysis included:

- Home size
- Lot size
- Bedrooms
- Full bathrooms
- Garage spaces
- Property age
- Pool indicator
- Postal code
- Crime rate
- Geographic and neighborhood characteristics

Additional data cleaning was performed to separate variables originally stored together, such as bathroom types and lot-size measurements.

Binary variables were also created for characteristics such as whether a property was detached and whether it contained a pool.

---

## Geospatial Feature Engineering

Additional geographic variables were engineered to capture the effect of location on property value.

Property addresses were first combined into complete address strings and then converted into latitude and longitude coordinates using Python geocoding.

Using **Python, GeoPandas, GIS shapefiles, and geographic distance calculations**, additional variables were created for:

- Distance to the nearest elementary school
- Distance to the nearest middle school
- Distance to the nearest high school
- Distance to the coastline
- Distance to the nearest downtown area

Average high and low temperatures were also incorporated based on the city in which each property was located.

### Haversine Distance

Geographic distances between coordinates were calculated using the Haversine formula:

$$
d =
2R
\arcsin
\left(
\sqrt{
\sin^2\left(\frac{\Delta \phi}{2}\right)
+
\cos(\phi_1)\cos(\phi_2)
\sin^2\left(\frac{\Delta \lambda}{2}\right)
}
\right)
$$

where:

- $R$ = radius of the Earth
- $\phi$ = latitude
- $\lambda$ = longitude
- $d$ = geographic distance between two locations

---

## Exploratory Data Analysis

Before fitting the predictive models, exploratory analysis was performed to understand the distributions and relationships within the dataset.

The analysis included:

- Summary statistics
- Price distributions
- Outlier analysis
- Correlation matrices
- Multicollinearity analysis
- Feature distributions

The housing-price distribution was strongly right-skewed due to a relatively small number of very high-priced properties.

Correlation matrices were also used to remove highly correlated variables before model training.

---

# Modeling Approach

## 1. Hedonic Pricing Model

The hedonic model assumes that the price of a property can be decomposed into the value associated with its individual characteristics.

The general model can be written as:

$$
P_i =
\beta_0
+
\beta_1 X_{i1}
+
\beta_2 X_{i2}
+
\cdots
+
\beta_k X_{ik}
+
\epsilon_i
$$

where:

- $P_i$ = price of property $i$
- $X_{ik}$ = property or location characteristic
- $\beta_k$ = estimated marginal contribution of the characteristic
- $\epsilon_i$ = unexplained component of price

A major advantage of the hedonic model is its interpretability. Each coefficient represents the estimated change in property price associated with a one-unit change in that characteristic, holding the remaining variables constant.

Examples of influential characteristics identified by the model included home size, bathrooms, bedrooms, age, and location-related characteristics.

### Hedonic Model Performance

The final hedonic regression achieved:

**R² ≈ 0.77**

Approximately **77% of the variation in housing prices** was explained by the selected features.

---

## 2. Decision Tree Regression

A Decision Tree predicts housing prices by recursively dividing the feature space through a sequence of binary splits.

The model provides a nonlinear alternative to the hedonic regression and allows interactions between characteristics to emerge without specifying them explicitly.

The dataset was divided into:

- **70% training data**
- **30% testing data**

### Decision Tree Performance

**R² ≈ 0.48**

The relatively low R² indicated that a single Decision Tree did not generalize as effectively as the other approaches.

---

## 3. Random Forest Regression

Random Forest extends the Decision Tree approach by constructing an ensemble of trees using different samples and subsets of features.

For regression, predictions from the individual trees are averaged:

$$
\hat{y}_i =
\frac{1}{B}
\sum_{b=1}^{B}
T_b(x_i)
$$

where:

- $B$ = number of trees
- $T_b(x_i)$ = prediction produced by tree $b$
- $\hat{y}_i$ = final Random Forest prediction

### Initial Random Forest Performance

**R² ≈ 0.70**

The Random Forest significantly outperformed the single Decision Tree, so further model development focused on improving the Random Forest.

---

# Random Forest Optimization

## Outlier Treatment

Exploratory analysis revealed extreme values across several variables.

To reduce their influence on model estimation, the upper tail of selected variables was filtered.

The updated analysis removed the **top 5% of values** from selected variables before retraining the model.

---

## Log Transformation

Housing prices exhibited substantial positive skew.

The target variable was therefore log-transformed before training:

$$
y_i^* = \log(P_i)
$$

After prediction, the outputs were converted back to the original price scale.

The transformation was intended to reduce the influence of extremely high-priced properties and improve the model's ability to learn relationships across the broader housing-price distribution.

---

## Categorical Encoding

Postal code was added to the improved Random Forest model as an additional location feature.

Because ZIP code is categorical rather than continuous, it was converted into numerical variables using **one-hot encoding**.

---

## Hyperparameter Optimization

`RandomizedSearchCV` was used to search across combinations of Random Forest hyperparameters.

Parameters evaluated included:

- Number of estimators
- Maximum tree depth
- Minimum samples required to split a node
- Maximum number of features considered at each split

The optimized model selected:

| Hyperparameter | Selected Value |
| --- | ---: |
| Number of Trees | 300 |
| Maximum Depth | 20 |
| Minimum Samples Split | 2 |

---

## Cross-Validation

Model stability and potential overfitting were evaluated using **5-fold cross-validation**.

The training dataset was divided into five subsets. During each iteration:

1. Four folds were used to train the model.
2. The remaining fold was used for validation.
3. The process was repeated until every fold had served as the validation set.

The mean cross-validation score was:

$$
R^2_{CV} \approx 0.718
$$

The cross-validation score was relatively close to the final test-set R² of 0.757, providing evidence that substantial overfitting was not present.

---

# Model Evaluation

## R-Squared

Model performance was primarily evaluated using the coefficient of determination:

$$
R^2 =
1 -
\frac{
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
}{
\sum_{i=1}^{n}
(y_i-\bar{y})^2
}
$$

where:

- $y_i$ = actual property price
- $\hat{y}_i$ = predicted property price
- $\bar{y}$ = mean property price

A larger $R^2$ indicates that the model explains a greater proportion of the variation in housing prices.

---

## Root Mean Squared Error

Prediction error was also evaluated using Root Mean Squared Error:

$$
RMSE =
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
}
$$

The optimized Random Forest achieved an RMSE of approximately:

**$161,171**

---

## Residual Analysis

Residuals were calculated as:

$$
e_i = y_i-\hat{y}_i
$$

Residual plots were used to examine whether prediction errors occurred randomly across the price distribution or whether systematic patterns remained.

The improved Random Forest produced residuals that were generally closer to zero than the initial model.

However, residual variance increased for higher-priced properties.

---

# Model Performance

| Model | R² |
| --- | ---: |
| Decision Tree | 0.48 |
| Initial Random Forest | 0.70 |
| Optimized Random Forest | **0.757** |
| Hedonic Regression | **0.77** |

### Optimized Random Forest

- **Test R²:** 0.757
- **RMSE:** ~$161,171
- **5-Fold Cross-Validation R²:** 0.718

The hedonic regression produced the highest overall R², while the Random Forest provided greater flexibility for modeling nonlinear relationships between property characteristics.

---

# Feature Importance

Feature importance from the optimized Random Forest was used to determine which variables contributed most strongly to predictions.

The most influential features included:

1. Home size
2. Lot size
3. Total crime rate
4. Property age
5. Bedrooms
6. Garage spaces
7. Postal-code indicators
8. Full bathrooms

**Home size was the dominant feature**, accounting for approximately 30% of the model's feature importance.

Several ZIP-code indicators also appeared among the most influential variables, suggesting that location contributed meaningful information beyond the physical characteristics of the property.

Notable postal codes appearing in the top features included areas corresponding to:

- Cambria and Harmony
- Pismo Beach
- Paso Robles and surrounding communities

---

# Prediction Error Analysis

A significant finding from the Random Forest analysis was its performance across different price ranges.

For properties near the center of the housing-price distribution, predictions generally tracked actual prices closely.

For higher-priced homes, however, the model increasingly tended to underpredict actual property values.

Residual variance also increased as home prices increased.

This suggests that the available features may not fully capture characteristics unique to expensive or luxury properties.

Potential missing variables could include:

- Luxury amenities
- Premium views
- Architectural characteristics
- Renovation quality
- Detailed neighborhood characteristics
- Other features specific to high-value homes

---

# Interactive Housing Map

In addition to the predictive models, an **interactive geographic visualization** was developed for properties throughout San Luis Obispo County.

Users can navigate through the county, zoom into individual neighborhoods, and select properties.

For each property, the map displays:

- Actual sale price
- Predicted sale price
- Prediction error / margin of error

The map provides a spatial representation of model performance and demonstrates a potential application of the analysis for real estate professionals.

Comparing predicted and actual values geographically can help identify properties that may appear relatively overvalued or undervalued according to the model.

---

# Key Findings

- Hedonic regression achieved the highest overall R² at approximately **77%**.
- The initial Decision Tree explained only approximately **48%** of housing-price variation.
- Random Forest substantially outperformed the single Decision Tree.
- The initial Random Forest achieved approximately **70% R²**.
- Model optimization increased Random Forest performance to **75.7% R²**.
- The optimized Random Forest achieved an RMSE of approximately **$161K**.
- The 5-fold cross-validation R² of **0.718** was reasonably close to test performance.
- Home size was the most important Random Forest feature.
- Lot size, crime rate, age, bedrooms, garage spaces, bathrooms, and location also contributed to predictions.
- Location information captured through postal codes provided additional predictive information.
- Prediction accuracy decreased for high-priced properties.
- Hedonic regression provided greater interpretability, while Random Forest offered greater flexibility for capturing nonlinear relationships.

---

# Methods Used

### Econometric Methods
- Hedonic pricing regression
- Correlation analysis
- Multicollinearity analysis
- Residual analysis

### Machine Learning
- Decision Tree Regression
- Random Forest Regression
- Randomized hyperparameter search
- 5-fold cross-validation
- Feature importance analysis
- Train/test evaluation

### Data Engineering
- Data cleaning
- Outlier filtering
- Log transformation
- One-hot encoding
- Geocoding
- Geographic feature engineering

### Geospatial Analysis
- Python
- GeoPandas
- GIS shapefiles
- Latitude/longitude coordinates
- Haversine distance calculations
- Interactive geographic visualization

---

# Future Research

Potential extensions of the project include:

- Testing Gradient Boosting models
- Incorporating additional luxury-home characteristics
- Adding more neighborhood-level variables
- Expanding geographic features
- Improving predictions for high-value properties
- Exploring additional machine learning approaches
- Further investigating systematic prediction errors

The residual analysis suggests that additional information specific to high-end properties could be particularly useful in improving future model performance.

---

# Conclusion

This project compared traditional econometric modeling with machine learning approaches for residential property valuation in San Luis Obispo County.

The **hedonic regression model achieved the highest overall R² of approximately 0.77**, demonstrating the effectiveness of an interpretable econometric approach for the available data.

At the same time, the optimized **Random Forest achieved an R² of 0.757**, substantially outperforming the single Decision Tree while capturing nonlinear relationships between property characteristics.

The project demonstrates how **econometrics, machine learning, data engineering, and geospatial analysis** can be combined to study housing valuation and build predictive tools for real estate applications.

---

## Author

**Matina S. Lampsas**

Orfalea College of Business  
California Polytechnic State University, San Luis Obispo

**BUS 464 – Applied Senior Project Seminar**  
December 2024
