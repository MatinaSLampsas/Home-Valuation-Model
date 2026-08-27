# Predicting Home Prices: Hedonic Regression & Machine Learning
Built and optimized housing price prediction models on 12.5K observations using Random Forest, geospatial feature engineering, cross-validation, and interactive GIS mapping, achieving 75.7% R².
A comparative housing valuation study using **hedonic regression, Decision Trees, and Random Forests** to predict residential property prices in San Luis Obispo County, California.

The project combines traditional econometric modeling with machine learning and geospatial feature engineering to examine the determinants of home prices and improve predictive performance across approximately **12,500 housing observations**.

## Project Overview

The objective of this research was to:

- Compare traditional hedonic regression with tree-based machine learning models
- Identify the property and location characteristics most influential in home valuation
- Engineer geographic features using GIS and geocoded property data
- Evaluate model performance using out-of-sample prediction and cross-validation
- Investigate model errors, particularly for high-value properties
- Build an interactive geographic visualization of actual and predicted home prices

## Data

The dataset contains approximately **12,500 residential property transactions** from San Luis Obispo County.

### Property Features

Features used throughout the analysis include:

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

### Geospatial Feature Engineering

Property addresses were converted into latitude and longitude coordinates through geocoding.

Additional geographic features were then constructed using **Python, GeoPandas, GIS shapefiles, and geographic distance calculations**, including:

- Distance to nearest elementary school
- Distance to nearest middle school
- Distance to nearest high school
- Distance to the coastline
- Distance to the nearest downtown area

Weather characteristics were also incorporated based on property location.

## Methodology

### 1. Hedonic Pricing Model

The hedonic model assumes that the price of a home can be decomposed into the values associated with its individual characteristics.

The general model can be represented as:

\[
P_i = \beta_0 + \beta_1 X_{i1} + \beta_2 X_{i2} + \cdots + \beta_k X_{ik} + \epsilon_i
\]

where:

- \(P_i\) = price of property \(i\)
- \(X_{ik}\) = property or location characteristic
- \(\beta_k\) = estimated marginal contribution of the characteristic
- \(\epsilon_i\) = unexplained component of price

Unlike the machine learning models, the hedonic regression allows the coefficients to be interpreted directly as the estimated change in price associated with a one-unit change in a feature, holding other variables constant.

The final hedonic model achieved an:

**R² ≈ 0.77**

meaning approximately 77% of the variation in housing prices was explained by the selected characteristics.

---

### 2. Decision Tree Regression

A Decision Tree recursively divides the feature space using binary splits and assigns predictions to the resulting terminal nodes.

The model provides a nonlinear alternative to hedonic regression and can capture interactions between property characteristics without specifying them beforehand.

Data was divided into:

- **70% training**
- **30% testing**

The Decision Tree achieved approximately:

**R² ≈ 0.48**

---

### 3. Random Forest Regression

Random Forest improves upon a single Decision Tree by constructing an ensemble of trees using different samples and subsets of features.

For regression, the final prediction is obtained by averaging predictions across the individual trees:

\[
\hat{y}_i =
\frac{1}{B}
\sum_{b=1}^{B}
T_b(x_i)
\]

where:

- \(B\) = number of trees
- \(T_b(x_i)\) = prediction from tree \(b\)
- \(\hat{y}_i\) = final Random Forest prediction

The initial Random Forest achieved:

**R² ≈ 0.70**

Further model development focused on improving this model.

## Random Forest Optimization

### Outlier Treatment

Initial exploratory analysis revealed extreme values in several variables.

To reduce their influence, observations in the upper tail of selected features were filtered, with the analysis examining values after removing the **top 5%**.

### Log Transformation

Housing prices exhibited substantial right skew.

The target variable was therefore transformed prior to model training:

\[
y_i^{*} = \log(P_i)
\]

Predictions were subsequently transformed back to the original dollar scale.

### Categorical Encoding

Postal code was incorporated as a location feature.

Because ZIP code is categorical, it was converted into numerical features using **one-hot encoding**.

### Hyperparameter Optimization

`RandomizedSearchCV` was used to search across Random Forest hyperparameters including:

- Number of estimators
- Maximum tree depth
- Minimum samples required to split
- Maximum number of features considered at each split

The optimized model included:

| Hyperparameter | Selected Value |
|---|---:|
| Number of Trees | 300 |
| Maximum Depth | 20 |
| Minimum Samples Split | 2 |

### Cross-Validation

Model stability was evaluated using **5-fold cross-validation**.

The training sample was divided into five subsets. The model was repeatedly trained on four folds and evaluated on the remaining fold.

Mean cross-validation score:

\[
R^2_{CV} \approx 0.718
\]

The similarity between the cross-validation score and final test performance suggested limited evidence of substantial overfitting.

## Model Evaluation

### R-Squared

Model performance was primarily evaluated using the coefficient of determination:

\[
R^2 =
1 -
\frac{
\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
}{
\sum_{i=1}^{n}(y_i-\bar{y})^2
}
\]

where a larger \(R^2\) represents a greater proportion of housing-price variation explained by the model.

### Root Mean Squared Error

Prediction error was also evaluated using RMSE:

\[
RMSE =
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
}
\]

### Residual Analysis

Residuals were calculated as:

\[
e_i = y_i-\hat{y}_i
\]

Residual plots were used to investigate systematic prediction errors and potential heteroskedasticity across different home-price ranges.

## Model Performance

| Model | R² |
|---|---:|
| Decision Tree | 0.48 |
| Initial Random Forest | 0.70 |
| Optimized Random Forest | **0.757** |
| Hedonic Regression | **0.77** |

Optimized Random Forest:

- **R²:** 0.757
- **RMSE:** ~$161,171
- **5-Fold CV R²:** 0.718

The hedonic model produced the highest overall R², while the Random Forest provided greater flexibility for modeling nonlinear relationships between housing characteristics.

## Feature Importance

Feature importance from the optimized Random Forest identified **home size** as the strongest predictor of home prices.

Other important variables included:

1. Home size
2. Lot size
3. Total crime rate
4. Property age
5. Bedrooms
6. Garage spaces
7. Postal-code indicators
8. Full bathrooms

Home size accounted for approximately **30% of Random Forest feature importance**.

Several ZIP-code indicators also appeared among the model's most important features, suggesting that location captured additional variation in home prices beyond physical property characteristics.

## Residual Analysis

Residual analysis revealed an important limitation of the model.

Prediction errors increased for properties in the upper portion of the price distribution, with the Random Forest tending to underpredict some expensive homes.

This suggests that the available dataset may not fully capture characteristics unique to luxury properties, such as:

- Premium views
- Architectural quality
- Renovations
- Luxury amenities
- Micro-neighborhood characteristics
- Waterfront or coastal desirability

## Interactive Housing Map

An interactive map was also developed to visualize predictions geographically across San Luis Obispo County.

Each property can be selected to display:

- Actual sale price
- Predicted price
- Prediction error

This provides a geographic representation of model performance and can help identify properties or regions where homes may appear relatively overvalued or undervalued according to the model.

## Geospatial Distance Calculation

Distances between geographic coordinates were calculated using geographic distance methods including the **Haversine formula**:

\[
d =
2R
\arcsin
\left(
\sqrt{
\sin^2\left(\frac{\Delta\phi}{2}\right)
+
\cos(\phi_1)\cos(\phi_2)
\sin^2\left(\frac{\Delta\lambda}{2}\right)
}
\right)
\]

where:

- \(R\) = radius of the Earth
- \(\phi\) = latitude
- \(\lambda\) = longitude
- \(d\) = distance between two geographic coordinates

This was used to construct proximity variables such as distance to downtown areas and other geographic points of interest.

## Tools & Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- GeoPandas
- GIS / Shapefiles
- Geocoding
- Matplotlib
- Statistical Regression
- Random Forest Regression
- Decision Tree Regression
- RandomizedSearchCV
- Cross-Validation
- Geographic Visualization

## Key Findings

- Traditional hedonic regression remained highly competitive, achieving an R² of approximately **77%**.
- Random Forest substantially outperformed a single Decision Tree.
- Hyperparameter tuning, target transformation, feature filtering, and additional location information improved Random Forest performance to **75.7% R²**.
- Home size was the most important predictor across the modeling analysis.
- Geographic and neighborhood variables contributed meaningfully to home valuation.
- Prediction accuracy deteriorated for high-value properties, suggesting that additional luxury-home characteristics could improve future models.
- Econometric and machine learning approaches provide complementary benefits: regression offers interpretability, while Random Forest captures more complex nonlinear relationships.

## Future Research

Potential extensions include:

- Gradient Boosting / XGBoost
- Additional luxury-property characteristics
- Neighborhood-level socioeconomic variables
- More detailed geographic features
- Spatial regression models
- Additional interaction effects
- SHAP-based model interpretation
- Improved treatment of high-value properties
- Expanded hyperparameter optimization

## Author

**Matina S. Lampsas**

California Polytechnic State University, San Luis Obispo  
Applied Senior Project Seminar
