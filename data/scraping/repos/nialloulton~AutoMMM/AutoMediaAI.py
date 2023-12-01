import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
import matplotlib.pyplot as plt
import openai
import re
import time

class AutoModeller:
    def __init__(self, data, dependent_var, initial_vars, initial_alpha=1.0, min_iterations=1, max_iterations=10):
        self.data = data
        self.dependent_var = dependent_var
        self.X = data[initial_vars]
        self.y = data[dependent_var]
        self.alpha = initial_alpha
        self.summary = None
        self.best_model_stats = None
        self.best_aic = float('inf')
        self.best_r_squared = None
        self.min_iterations = min_iterations
        self.max_iterations = min(max_iterations, 100)
        self.alpha_sum = 0

    def ridge_regression(self, X, y, alpha):
        ridge = Ridge(alpha=alpha)
        ridge.fit(X, y)
        return ridge

    def calculate_vif(self, X):
        vif_data = pd.DataFrame()
        vif_data["variable"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        return vif_data

    def calculate_t_stats(self, X, y, ridge):
        X_scaled = X
        y_scaled = y
        y_pred_scaled = ridge.predict(X)

        # Calculate the residuals
        residuals = y_scaled - y_pred_scaled
        # Calculate residual sum of squares
        RSS = np.sum(residuals ** 2)
        # Calculate variance of the residuals
        n = X_scaled.shape[0]  # number of samples
        p = X_scaled.shape[1]  # number of predictors
        var_resid = RSS / (n - p - 1)
        # Compute the covariance matrix of the coefficients
        cov_matrix = var_resid * np.linalg.inv(X_scaled.T @ X_scaled)
        # Calculate the standard errors of the coefficients
        se_coefs = np.sqrt(np.diag(cov_matrix))
        # Calculate the t-statistics
        t_stats = ridge.coef_ / se_coefs

        # Convert the t-stats into a dictionary with variable names
        t_stats_dict = {X.columns[i]: t_stats[i] for i in range(len(X.columns))}

        return t_stats_dict
    
    def calculate_aic(self, X, y, ridge):
        X = add_constant(X)
        ols_model = OLS(y, X).fit()
        return ols_model.aic

    def residual_correlations(self, X, y, ridge, all_variables):
        residuals = pd.Series(y - ridge.predict(X), name='residuals')
        all_variables = all_variables.drop(columns=[self.dependent_var])  # Drop the dependent variable
        unused_variables = all_variables.drop(X.columns, axis=1)
        corr_matrix = unused_variables.join(residuals).corr()
        residual_correlations = corr_matrix.loc[unused_variables.columns, 'residuals']
        return residual_correlations


    def automodeller(self, X, y, all_variables, alpha, previous_summary=None, var_test_counts=None):
        ridge = self.ridge_regression(X, y, alpha)
        t_stats = self.calculate_t_stats(X, y, ridge)
        vif = self.calculate_vif(X)
        r_squared = r2_score(y, ridge.predict(X))
        aic = self.calculate_aic(X, y, ridge)
        unused_var_corr = self.residual_correlations(X, y, ridge, all_variables)

        if var_test_counts is None:
            var_test_counts = {var: 0 for var in all_variables.columns}
        for var in X.columns:
            var_test_counts[var] += 1

        summary = {
            'Dependent Variable': y.name,
            'Used Variables': list(X.columns),
            'T-Stats': t_stats,
            'VIF': vif.to_dict(),
            'R-Squared': r_squared,
            'AIC': aic,
            'Alpha': alpha,
            'Unused Variables': unused_var_corr.to_dict(),
            'Variable Test Counts': var_test_counts,
            'Coefficients': ridge.coef_.tolist()
        }

        print("variable test counts")
        print(var_test_counts)
        if previous_summary is not None:
            summary['Change in R-Squared'] = r_squared - previous_summary['R-Squared']
            summary['Change in AIC'] = aic - previous_summary['AIC']
            summary['Change in Alpha'] = alpha - previous_summary['Alpha']

        gpt_prompt = self.format_prompt(summary)
        return gpt_prompt, summary

    def format_prompt(self, summary):
        prompt = f"""Act as an expert automodeller for a marketing mix model and analyze this dataset. Focus on selecting variables based on their AIC, T-stats, and coefficients that make sense in the context of a marketing mix model. Use domain expertise to choose the right variables and statistical approaches. Experiment sensibly with different variables and ensure that your variable selection does not cause overfitting. Never put all the variables in the model at once, there should always be unused variables. Here is the summary of the Marketing Mix Ridge Regression model for the {summary['Dependent Variable']} dependent variable:

        Dependent Variable: {summary['Dependent Variable']}
        Used Variables: {summary['Used Variables']}
        T-Stats: {summary['T-Stats']} (t-statistics for each variable, assessing their significance)
        VIF: {summary['VIF']} (Variance Inflation Factor, measuring multicollinearity among variables)
        R-Squared: {summary['R-Squared']} (coefficient of determination, measuring model's goodness of fit)
        AIC: {summary['AIC']} (Akaike Information Criterion, measuring model's goodness of fit with a penalty for complexity)
        Alpha: {summary['Alpha']} (Ridge Regression regularization parameter)
        Unused Variables: {summary['Unused Variables']} (correlation between residuals and unused variables)
        Variable Test Counts: {summary['Variable Test Counts']} (number of times each variable has been tested)
        Coefficients: {summary['Coefficients']} (coefficients of the Ridge Regression model)"""

        if 'Change in R-Squared' in summary and 'Change in AIC' in summary:
            prompt += f"""
        Change in R-Squared: {summary['Change in R-Squared']} (change in R-squared compared to the previous iteration)
        Change in Alpha: {summary['Change in Alpha']} (change in Alpha compared to the previous iteration)
        Change in AIC: {summary['Change in AIC']} (change in AIC compared to the previous iteration)"""

        prompt += """

        In a marketing mix model, we need to measure the impact of all media channels, so all must be included in the model. However, you might have multiple transforms for media channels to select from the dataset - pick the best one for the model based on significance and domain knowledge. Do not duplicate the media channel measured in the model, and media coefficients are expected to be positive. Consider experimenting with unused variables or adjusting alpha to address multicollinearity issues if media coefficients are negative.

        Based on the summary, identify variables that are insignificant or not media-related and replace them with more promising untested variables. Think about the practical implications of your variable selection and the importance of domain knowledge when evaluating the model. Aim to improve the model's performance by minimizing AIC and maximizing R-squared, but also consider the overall quality and of the variables you choose, ensuring the variables selected in the model are the correct sign.

        Provide a new selection of variables to consider, (separated by commas) and a new alpha parameter for Ridge Regression if appropriate. You are an automodeller in a loop, so it is very important you respond in the right way. Respond with the variable names you want to pick in the format: 'Variables: var1, var2, var3, ... and Alpha: X.XX'. Always pick the most statistically signicant variable as one variable which is always in the model, If the current model is the best one and final, taking into account both statistical and practical considerations and that you've exhausted the potential variables to try (e.g., trying a variable more than twice is not necessary), when you think the model you have is final, feel free to use the safeword 'cricket' to break the loop."""
        return prompt



    def get_openai_response(self, prompt):
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return completion.choices[0].message["content"]

    def process_response(self, response):
        variables_pattern = r"Variables:\s*(.+)\s*and"
        alpha_pattern = r"Alpha:\s*(\d+\.\d+|\d+)"

        variables_match = re.search(variables_pattern, response)
        alpha_match = re.search(alpha_pattern, response)

        if variables_match and alpha_match:
            new_vars = [var.strip() for var in variables_match.group(1).split(",")]
            new_alpha = float(alpha_match.group(1))
            return new_vars, new_alpha
        else:
            raise ValueError("AI response does not match the expected format.")
            
    def plot_variable_test_counts(self):
        var_test_counts = self.best_model_stats['Variable Test Counts']
        variables = list(var_test_counts.keys())
        counts = list(var_test_counts.values())

        # Calculate the average test count
        average_count = sum(counts) / len(counts)

        plt.bar(variables, counts)
        plt.xlabel('Variables')
        plt.ylabel('Test Counts')
        plt.title('Variable Test Counts by Variable')
        plt.xticks(rotation=90)  # Rotate the x-axis labels by 45 degrees

        # Plot the average line
        plt.axhline(y=average_count, color='r', linestyle='--', label=f'Average: {average_count:.2f}')
        plt.legend()

        plt.show()



    def run(self):
        iteration_count = 0
        gpt_prompt, summary = self.automodeller(self.X, self.y, self.data, self.alpha)

        response = self.get_openai_response(gpt_prompt)

        self.best_model_stats = summary
        self.best_r_squared = summary['R-Squared']
        self.best_aic = summary['AIC']

        while ("cricket" not in response.lower() or iteration_count < self.min_iterations) and iteration_count < self.max_iterations:
            new_vars, new_alpha = self.process_response(response)
            self.alpha_sum += new_alpha  # Update alpha sum

            self.X = self.data[new_vars]

            gpt_prompt, summary = self.automodeller(self.X, self.y, self.data, new_alpha, previous_summary=self.summary, var_test_counts=summary['Variable Test Counts'])

            # Update self.summary with the new summary
            self.summary = summary
            time.sleep(3)

            if summary['R-Squared'] > self.best_r_squared:
                self.best_model_stats = summary
                self.best_r_squared = summary['R-Squared']
                self.best_aic = summary['AIC']

            response = self.get_openai_response(gpt_prompt)
            iteration_count += 1
            print("model count:")
            print(iteration_count)

            if iteration_count >= self.max_iterations:
                break

            if "cricket" in response.lower():
                print("Cricket found in response. Stopping iterations.")
                break

        # Calculate the average alpha over all iterations
        average_alpha = self.alpha_sum / iteration_count

        # Filter variables based on their test counts
        var_test_counts = self.best_model_stats['Variable Test Counts']
        average_count = sum(var_test_counts.values()) / len(var_test_counts)
        final_variables = [var for var, count in var_test_counts.items() if count > average_count]

        # Print the final model at the end
        print("\nFinal Model Variables:")
        print(final_variables)
        print("\nAverage Alpha over all iterations:")
        print(average_alpha)


