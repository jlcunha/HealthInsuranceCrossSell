import pandas           as pd
import numpy            as np
import seaborn          as sns
from matplotlib         import pyplot as plt

class asdas:
    """Class to evaluate insurance profit predictions.

    Parameters
    ----------
    data : pandas.DataFrame
        A pandas DataFrame containing the insurance scores predictions
        as well as the true labels. It should have the following columns:
        'scores' (float), 'label' (int).

    self.insurance_price : float
        The price of the insurance product.

    cost : float
        The cost of each deal.

    Attributes
    ----------
    data : pandas.DataFrame
        A copy of the input data DataFrame.

    insurance_price : float
        The price of the insurance product.

    cost : float
        The cost of each deal.
    """
    def __init__(self, data, insurance_price, cost):
        self.data = data
        self.insurance_price = insurance_price
        self.cost = cost
        

    def recall_at_k( self, data, k=2000):
        """
        Calculates the recall at k for the given data.

        Parameters:
        data (pandas.DataFrame): The data to evaluate.
        k (int): The value of k for recall at k.

        Returns:
        float: The recall at k for the given data.
        """
        # reset index
        data = data.reset_index(drop=True)

        # create ranking order
        data['ranking'] = data.index + 1

        # calculate the number of positive instances
        num_positives = data['response'].sum()

        # calculate cumulative sum of positive instances
        data['cumulative_positives'] = data['response'].cumsum()

        # calculate recall at k
        recall_at_k = data.loc[k, 'cumulative_positives'] / num_positives

        return recall_at_k


    def profit_dataframe(self):
        """
        Compute the profit and baseline for different fractions of the dataset.


        Returns
        -------
        pandas.DataFrame
            A dataframe with three columns:
            - 'x_nsamples': the fraction of the dataset used for computing the profit and baseline, from 1% to 99%.
            - 'y_profit': the profit obtained by selling insurance policies to the selected customers.
            - 'y_baseline': the profit obtained by selecting customers randomly.

        """

        number_of_positives = self.data['response'].sum()
        population_size = self.data.shape[0]
        k_numbers = np.arange(self.data.shape[0] // 100, self.data.shape[0], self.data.shape[0] // 100)
        k_percents = np.round((k_numbers / self.data.shape[0]), 2)

        recalls = np.array([self.recall_at_k(self.data, k=k_number) for k_number in k_numbers])
        profits = (recalls * number_of_positives * self.insurance_price) - (k_percents * population_size * self.cost)
        baselines = (k_percents * number_of_positives * self.insurance_price) - (k_percents * population_size * self.cost)

        self.df_profit = pd.DataFrame({
            'x_nsamples': k_percents,
            'y_profit': np.round(profits, 2),
            'y_baseline': np.round(baselines, 2)
        })
        
        return self.df_profit
    

    def line_profit(self, title='Best K for the Biggest Profit'):
        """
        Plots the profits obtained by a model and a baseline for different percentages of a sample, highlighting the point
        with the maximum profit for the model. 

        Args:
            title (str, optional): The title of the plot. Defaults to 'Best K for the Biggest Profit'.

        Returns:
            ax (matplotlib.axes.Axes): The axes object containing the plot.
        """

        plt.figure(figsize=(12, 6))

        # Get data for plot
        max_profit = self.df_profit.loc[self.df_profit['y_profit'].idxmax()]
        x_value, y_value = max_profit[['x_nsamples', 'y_profit']].astype(float)
        y_min_value = int(self.df_profit['y_profit'].min() * 1.1)
        y_baseline = int(max_profit['y_baseline'])

        # Create text strings
        text = f'Percentage of sample = {x_value}\nProfit = {y_value}'
        text_baseline = f'Percentage of sample = {x_value}\nProfit = {y_baseline}'

        # Plot the baseline
        ax = sns.lineplot(x='x_nsamples', y='y_baseline', data=self.df_profit, marker='o', fillstyle='none',
                          markeredgewidth=1, markeredgecolor='black', markevery=[max_profit.name], label='Baseline')

        # Plot the model
        sns.lineplot(x='x_nsamples', y='y_profit', data=self.df_profit, marker='o', fillstyle='none', markeredgewidth=1,
                     markeredgecolor='black', markevery=[max_profit.name], ax=ax, label='With Model')

        # Set plot information
        ax.set_xlabel('Percentage of sample')
        ax.set_ylabel('Profit')
        ax.set_title(title)
        ax.set_ylim([y_min_value, y_value * 1.5])
        ax.text(x_value, (y_value * 1.2), text, horizontalalignment='center', verticalalignment='center', fontsize=10)
        ax.text(x_value, (y_baseline + (((0.2 * y_baseline) ** 2) ** (1 / 2))), text_baseline, horizontalalignment='center', 
                verticalalignment='center', fontsize=10)

        plt.tight_layout()

        return ax

    def profit_bar(self, title='Profit: Selected Customers X All Customers'):
        """
        Plots a bar chart comparing the profits of selected customers versus all customers.

        Parameters
        ----------
        title : str, optional
            The title of the plot. Default is 'Profit: Selected Customers X All Customers'.

        Returns
        -------
        matplotlib.axes.Axes
            The plot of the profits.
        """
        y_value = np.max(self.df_profit['y_profit'])

        interested_customers = np.sum(self.data['response'])
        revenue = interested_customers * self.insurance_price
        all_cost = self.data.shape[0] * self.cost
        profit = revenue - all_cost

        plt.figure(figsize=(12,6))
        bar_plot = sns.barplot(x=['Selected Customers', 'All Customers'], y=[y_value, profit])
        bar_plot.set_title(title)
        bar_plot.set_ylabel('Profit')

        for rect, val in zip(bar_plot.patches, [y_value, profit]):    
            perc = np.round((val/profit*100), 2)
            bar_plot.annotate(f"{perc}% ({val})", (rect.get_x() + rect.get_width() / 2., rect.get_height()),
                         ha='center', va='center', fontsize=10, color='black', xytext=(0, 10),
                         textcoords='offset points')

        plt.tight_layout()

        return bar_plot