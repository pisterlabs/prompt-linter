#!/bin/python3

import decimal as d  # Allows me to accurately round up
from os import path  # Allows me to determine if file exists
from sys import exit as terminate  # Allows me to end with codes
from rich import print  # Allows me to color and format text output
from rich.console import Console  # Allows me to have clean word wrapping
import questionary


# ============ FUNCTIONS ============
def info(content, kind='info'):
    """
    This prints info to the terminal in a fancy way.

    :param content: This is the string you want to display.
    :param kind: 'bad', 'info', 'question'; changes color.
    :return: None
    """

    if kind == 'info':
        print(f'[bold blue]ℹ️[/bold blue][reset] [bold grey50]{content}[/bold grey50]')


    elif kind == 'bad':
        print(f'[bold red][[X]][/bold red] [black]{content}[/black]')

    elif kind == 'question':
        print(f'[bold yellow][[?]][/bold yellow] [black]{content}[/black]')
        

def paragraph(content: str, style='grey50'):
    """
    This prints properly wrapped text to the console.

    :param content: a string that you want printed
    :return: None
    """
    console = Console(width=65)
    console.print(f'{content}', style=style, justify='left')


def getDecimal(rawFloat: float):
    """
    This function will convert any passed floats into a decimal to the hundredths place.

    :param rawFloat: This is an unconverted float
    :return: Float object of hundredths place
    """

    dVal = d.Decimal  # Create decimal object
    cent = dVal('0.01')  # Create placeholder
    converted = float(dVal(str(rawFloat)).quantize(cent, rounding=d.ROUND_UP))  # Convert to quantized d then to float
    return converted  # Return final product


def getExpenses():
    """
    This function will either get expenses from file in the format of "Expense,Cost"

    :return: list of tuples, where every element has (string: item name,
                                                      float: item cost)
    """

    # 2. Open file and get expenses
    with open('expenses.acct', 'r') as file:
        expenses = file.read().split('\n')

    # 3. Convert to appropriate list
    itemCosts = []

    for expense in expenses:
        try:
            itemCost = expense.split(',')

            if not itemCost[0] == '':
                itemCosts.append((itemCost[0], float(itemCost[1])))

        except Exception as e:  # Stop if format is wrong
            error = str(e)
            info('Failure: Expenses File Format Wrong', kind='bad')
            info(f'Error Cause: {error}')  # This should only appear if the exception was triggered
            info('Ensure that the cost section has no non-numeral characters other than a period.')
            terminate(4)

    return itemCosts


def getExpenseSum(itemCosts: list):
    """
    This function calculates the sum of all recurring expenses and returns a decimal

    :param itemCosts: list of (item, cost) tuple
    :return: float decimal
    """

    # 1. Set Up Display and Total
    print('[bold red]========= EXPENSES =========[/bold red]')

    totalExpense = 0

    # 2. Add to total for each item in expenses
    for pair in itemCosts:  # For each item
        totalExpense += pair[1]  # [1] is the numerical value
        info(f'Reserved ${pair[1]:.02f} for {pair[0]}')

    return getDecimal(totalExpense)


def getBudget(paycheck: float, expenses: float, save: int, invest: int):
    """
    This function divides your available cash into several portions,
    depending on how you wish to allocate it.

    :param paycheck: This is how much money you have to use
    :param expenses: This is how much money you must spend
    :param save: This represents a percentage of your budget to save
    :param invest: This represents a percentage of your budget to invest
    :return: None
    """

    # 1. Determine if okay to continue onto the computation
    if (save + invest) > 100:  # If you have a bad percentage combination
        info('Invalid Allocation of Available Funds', kind='bad')
        info('You may have it so that the percent you save and invest is ' +
             'greater than 100%\n')
        terminate(5)

    elif expenses > paycheck:  # If you're spending more than you have
        print('\n[bold red]========= BUDGET =========[/bold red]')
        info('WARNING: BUDGET DEFICIT', kind='bad')
        info('Your paycheck was insufficient to meet your expenses.\n' +
             'As a consequence, you have a net loss this period.')
        info(f'Current Deficit -------> [bold italic underline red]' +
             f'${getDecimal(paycheck - expenses):.02f}[/bold italic underline red]\n')
        terminate(0)  # This represents a successful run of the script

    # 2. Compute budget
    free = paycheck - expenses
    save = getDecimal((save / 100) * free)
    invest = getDecimal((invest / 100) * free)
    spend = getDecimal(paycheck - (save + invest + expenses))

    # 3. Inform user
    print('\n[bold green]========= BUDGET =========[/bold green]')
    info(f'You will pay ${expenses:.02f} for your expenses')
    info(f'You will invest ${invest:.02f}')
    info(f'You will save ${save:.02f}')
    info(f'Available Budget -------> [bold green]${spend:.02f}[/bold green]\n')
    # 4. END
    terminate(0)

def setupConfig():
    """
    This determines if you set up the files needed

    :return: None
    """

    # Exit variable
    end = False

    # 1. Determine if expenses file exists
    if not path.isfile('expenses.acct'):
        info('Failed to Locate Expenses File', kind='bad')

        with open('expenses.acct', 'w') as file:  # This creates a file with example
            file.write('example,0.00')

        paragraph('Made expenses.acct file for you. Please enter expenses in the format "expense,x.xx" ' +
                  'without the quote, where "x.xx" represents a decimal. Create a new line for every expense.\n')
        end = True

    # 2. Determine if the config file exists
    if not path.isfile('config.acct'):
        info('Failed to Locate Config File', kind='bad')

        # Create config file
        with open('config.acct', 'w') as file:
            file.write('Save:0\nInvest:0')
            paragraph('Created config.acct file for you. The please replace the "0" ' +
                      'with a whole number representing a percentage. This value will tell ' +
                      'me how to allocate your budget.\n')
            end = True

            # End?
    if end:
        terminate(3)


def getConfig():
    """
    Read from the config file. Terminate if bad format.

    :return: tuple containing int, int for save, invest
    """

    # 1. Open file and determine if settings are in there
    with open('config.acct', 'r') as file:
        content = file.read()
        if not 'Invest' in content and not 'Save' in content:
            info('Config file not formatted properly', kind='bad')
            terminate(6)

        else:
            options = content.split('\n')

    # 2. Find option values
    gotInvest = False  # Represents if invest value was found
    gotSave = False  # Both of these need to turn True to pass

    for option in options:  # For every located option in config
        if 'Invest' in option:  # If you find the invest option
            invest = option.split(':')[1]  # Get its numerical value
            gotInvest = True  # Flip its switch to True

        elif 'Save' in option:  # Do the same for save
            save = option.split(':')[1]
            gotSave = True

        if gotInvest and gotSave:  # If both switches were flipped
            try:
                return (int(save), int(invest))  # Return the information

            except Exception as e:  # Otherwise tell me what's up
                info('Unknown Error:' + str(e), kind='bad')
                terminate(9)

    terminate(7)  # End program: config file not formatted properly
def addExpense(new_expense):
    """
    Adds a new expense to the expenses.acct file in the format "name,amount".

    :param new_expense: The new expense to add in the format "name,amount".
    :return: None
    """

    # 1. Validate the new expense format
    try:
        name, amount = new_expense.split(',')
        amount = getDecimal(float(amount))  # Convert amount to decimal
    except ValueError:
        info('Invalid expense format. Use "name,amount".', kind='bad')
        terminate(8)

    # 2. Add to the expenses.acct file
    with open('expenses.acct', 'a') as file:
        file.write(f'{name},{amount:.2f}\n')
    info(f'Added expense: {name} with amount ${amount:.2f}', kind='info')

def deleteExpense(expense_name):
    """
    Deletes an expense from the expenses.acct file.

    :param expense_name: The name of the expense to delete.
    :return: None
    """

    # 1. Open the expenses.acct file
    with open('expenses.acct', 'r') as file:
        expenses = file.read().split('\n')

    # 2. Find the expense to delete
    for expense in expenses:
        if expense_name in expense:
            expenses.remove(expense)

    # 3. Write the new expenses to the file
    with open('expenses.acct', 'w') as file:
        for expense in expenses:
            file.write(f'{expense}\n')

    info(f'Deleted expense: {expense_name}', kind='info')
    
def purchase_suggest_gpt(expenses, cost):
    from openai import OpenAI
    MODEL = "gpt-3.5-turbo"
    response = OpenAI(api_key=settings['openai_api_key']).chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful accountant that will tell me if my next purchase is recommendable or not, considering my financial situation. You will only answer with 'Affordable' or 'Not Affordable', followed by a coincise explanation."},
            {"role": "user", "content": f"Given the following expenses, tell me if I can afford a new purchase of amount {cost}. My expenses are: {expenses}, my monthly income is {settings['monthly_income']}, my monthly rent is {settings['monthly_rent']}, my current savings are {settings['current_savings']}. Answer only with 'Affordable' or 'Not Affordable', followed by a coincise explanation."},
        ],
        temperature=0,
    )
    return response.choices[0].message.content

def optimize_budget_gpt(expenses):
    from openai import OpenAI
    MODEL = "gpt-3.5-turbo"
    response = OpenAI(api_key=settings['openai_api_key']).chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful accountant, tasked with optimizing my budget. Answer with the amount of money I should allocate to each category of my expenses or how I should invest my funds, given my monthly income, monthly rent, and current savings. You will only answer with a number, followed by a coincise explanation."},
            {"role": "user", "content": f"Given the following parameters, tell me how to optimize my budget. My expenses are: {expenses}, my monthly income is {settings['monthly_income']}, my monthly rent is {settings['monthly_rent']}, my current savings are {settings['current_savings']}."},
        ],
        temperature=0,
    )
    return response.choices[0].message.content

# ============ MENU ============
import plotext as plt

def cli_plot_expenses(itemCosts):
    """
    Plots expenses as a CLI bar chart using plotext.

    :param itemCosts: list of tuples with (item, cost)
    :return: None
    """
    items = [item for item, cost in itemCosts]
    costs = [cost for item, cost in itemCosts]

    # Smaller labels for better display


    plt.bar(items, costs)
    plt.title("Expenses Bar Chart")
    plt.xlabel("Items")
    plt.ylabel("Cost ($)")
    plt.show()

def display_statistics(itemCosts):
    """
    Calculates and displays statistics of expenses.

    :param itemCosts: list of tuples with (item, cost)
    :return: None
    """
    # Calculate statistics
    total_expenses = sum(cost for _, cost in itemCosts)
    average_expense = total_expenses / len(itemCosts) if itemCosts else 0
    max_expense = max(itemCosts, key=lambda x: x[1], default=('None', 0))
    min_expense = min(itemCosts, key=lambda x: x[1], default=('None', 0))

    # Display statistics
    print('\n[bold magenta]========= STATISTICS =========[/bold magenta]')
    info(f'Total Expenses: ${total_expenses:.2f}')
    info(f'Average Expense: ${average_expense:.2f}')
    info(f'Highest Expense: {max_expense[0]} - ${max_expense[1]:.2f}')
    info(f'Lowest Expense: {min_expense[0]} - ${min_expense[1]:.2f}')

    # If you want to categorize expenses and show stats per category, you would need
    # to modify your expense tracking to include categories and adjust the code accordingly.
    print()
    show_plot = questionary.confirm("Would you like to see a bar chart of your expenses?").ask()
    if show_plot:
        cli_plot_expenses(itemCosts)
        input("\nPress enter to continue...")

def affordability_metrics(income, expenses, debts, current_savings, computer_cost, potential_income_increase):
    """
    Calculate various financial metrics to determine if purchasing a computer is affordable.

    :param income: Monthly net income.
    :param expenses: Monthly expenses.
    :param debts: Monthly debt payments.
    :param current_savings: Current total savings.
    :param computer_cost: Cost of the computer.
    :param potential_income_increase: Potential monthly income increase due to the computer.
    :return: Dictionary of affordability metrics and a decision based on them.
    """
    metrics = {
        'dti_ratio': (debts / income) * 100,  # Debt-to-Income Ratio
        'savings_ratio': ((income - expenses - debts) / income) * 100,  # Savings Ratio
        'expense_ratio': (expenses / income) * 100,  # Expense-to-Income Ratio
        'liquidity_ratio': current_savings / expenses,  # Liquidity Ratio
        'roi': (potential_income_increase * 12) / computer_cost  # Return on Investment
    }

    # Decision rules based on metrics
    decision = "Affordable" if (metrics['dti_ratio'] < 35 and
                                metrics['savings_ratio'] > 20 and
                                metrics['expense_ratio'] < 50 and
                                metrics['liquidity_ratio'] > 6 and
                                metrics['roi'] > 1) else "Not Affordable"

    return metrics, decision

def detailed_affordability_metrics(income, expenses, debts, current_savings, computer_cost, potential_income_increase,
                                   emergency_fund_months=6, debt_interest_rate=0, computer_lifespan_years=5, financing_rate=0):
    """
    Calculate various financial metrics to determine if purchasing a computer is affordable, providing detailed explanations
    and suggestions based on the financial health indicators and treating ROI as a supplementary metric.

    :param income: Monthly net income.
    :param expenses: Monthly expenses.
    :param debts: Monthly debt payments.
    :param current_savings: Current total savings.
    :param computer_cost: Cost of the computer.
    :param potential_income_increase: Potential monthly income increase due to the computer.
    :param emergency_fund_months: The number of months of expenses that should be available in an emergency fund.
    :param debt_interest_rate: The annual interest rate on current debts.
    :param computer_lifespan_years: The expected lifespan of the computer in years.
    :param financing_rate: The annual interest rate if the computer is to be financed.
    :return: Dictionary of affordability metrics, a decision based on them, and detailed explanations and suggestions.
    """
    def try_divide(numerator, denominator):
        """
        Try to divide two numbers, returning 0 if the denominator is 0.

        :param numerator: The numerator.
        :param denominator: The denominator.
        :return: The result of the division or 0 if the denominator is 0.
        """
        return numerator / denominator if denominator else 0

    # Convert string input to floats
    monthly_income = float(income)
    monthly_expenses = float(expenses)
    monthly_debts = float(debts)

    current_savings = float(current_savings)
    computer_cost = float(computer_cost)
    potential_income_increase = float(potential_income_increase)

    # Calculate affordability metrics
    if any(value < 0 for value in [monthly_income, monthly_expenses, monthly_debts, current_savings, computer_cost, potential_income_increase]):
                print("All values should be positive. Please try again.")
                input("\nPress enter to continue...")
                return

    ideal_savings_amount = 10000
    # Calculate financial metrics
    metrics = {
        'dti_ratio': (try_divide(debts, income) * 100),  # Debt-to-Income Ratio
        'savings_ratio': try_divide((income - expenses - debts), income) * 100,  # Savings Ratio
        'expense_ratio': try_divide(expenses, income) * 100,  # Expense-to-Income Ratio
        'liquidity_ratio': try_divide(current_savings, expenses * emergency_fund_months),  # Liquidity Ratio
        'roi': try_divide((potential_income_increase * 12), computer_cost) if potential_income_increase else 0,  # Return on Investment
        'flexibility_ratio': try_divide(1 - try_divide(expenses + debts, income), 1) * 100,  # Financial Flexibility Ratio
        'interest_paid_on_debt': debts * (try_divide(debt_interest_rate, 100)),  # Monthly interest on current debt
        'total_cost_with_financing': computer_cost * (try_divide(1 + financing_rate, 100)),  # Total cost if financed
        'break_even_months': try_divide(computer_cost, potential_income_increase) if potential_income_increase else float('inf'),  # Months to break even on investment
        'ideal_savings_respected': try_divide((current_savings - computer_cost), ideal_savings_amount) * 100  # Ideal savings respected
    }

    metrics['depreciated_roi'] = ((potential_income_increase * 12 * computer_lifespan_years) /
                                  metrics['total_cost_with_financing']) if potential_income_increase else 0  # ROI considering depreciation and financing

    # Initialize decision and explanation
    decision = "Affordable"
    explanation = []
    suggestions = []

    # Evaluate affordability based on metrics
    if metrics['dti_ratio'] >= 35:
        decision = "Not Affordable"
        explanation.append(f"Your debt-to-income ratio is {metrics['dti_ratio']:.2f}%, which is above the healthy threshold of 35%.")
        suggestions.append("Consider paying down some debts to lower your DTI ratio before making large purchases.")

    if metrics['savings_ratio'] <= 20:
        decision = "Not Affordable"
        explanation.append(f"Your savings ratio is {metrics['savings_ratio']:.2f}%, which is below the recommended 20% for financial resilience.")
        suggestions.append("Try to increase your monthly savings to improve your savings ratio.")

    if metrics['expense_ratio'] >= 50:
        decision = "Not Affordable"
        explanation.append(f"Your expense-to-income ratio is {metrics['expense_ratio']:.2f}%, leaving little room for additional expenses.")
        suggestions.append("Look for ways to reduce your monthly expenses to improve your expense ratio.")

    if metrics['liquidity_ratio'] <= 1:
        decision = "Not Affordable"
        explanation.append("You have just enough savings to cover your expenses for the recommended emergency fund duration.")
        suggestions.append("It's advisable to build a larger financial cushion before incurring additional costs.")

    if metrics['flexibility_ratio'] <= 20:
        decision = "Not Affordable"
        explanation.append(f"Your financial flexibility ratio is {metrics['flexibility_ratio']:.2f}%, indicating limited discretionary income.")
        suggestions.append("Increase your financial flexibility by increasing income or decreasing non-discretionary expenses.")

    if metrics['depreciated_roi'] <= 1 and potential_income_increase:
        explanation.append("Although ROI is not the primary decision factor, it's worth noting that the return on investment is low when considering the computer's depreciation.")
    if potential_income_increase:
        if metrics['break_even_months'] >= (computer_lifespan_years * 12):
            explanation.append("The time to break even on the computer purchase is longer than the expected lifespan of the computer, which is not ideal financially.")
    if metrics['ideal_savings_respected'] <= 100:
        explanation.append(f"You have ${current_savings - computer_cost:.2f} less than the ideal savings amount of ${ideal_savings_amount:.2f}.")
        suggestions.append("Consider saving more before making the purchase.")
        decision = "Not Affordable"

    # Compile the detailed report
    detailed_report = " ".join(explanation + suggestions)
    return metrics, decision, detailed_report

def ask_for_financial_metrics():
        # Gather financial information using questionary
    monthly_income = settings['monthly_income']
    monthly_expenses = 0

    with open('expenses.acct', 'r') as file:
        for line in file:
            if line.strip():  # Check if the line is not empty
                _, amount = line.split(',')  # Split each line into category and amount
                monthly_expenses += float(amount.strip())  # Convert amount to float and add to total
    if monthly_expenses is None:
        monthly_expenses = questionary.text("What are your total monthly expenses?", validate=lambda text: text.replace('.', '', 1).isdigit()).ask()
    monthly_debts = questionary.text("What are your total monthly debt payments?", validate=lambda text: text.replace('.', '', 1).isdigit()).ask()
    current_savings = settings['current_savings'] if settings['current_savings'] else questionary.text("What is your current total savings?", validate=lambda text: text.replace('.', '', 1).isdigit()).ask()
    computer_cost = questionary.text("What is the cost of the product you want to buy?", validate=lambda text: text.replace('.', '', 1).isdigit()).ask()
    potential_income_increase = questionary.text("What is the potential monthly income increase?", validate=lambda text: text.replace('.', '', 1).isdigit()).ask()
    return float(monthly_income), float(monthly_expenses), float(monthly_debts), float(current_savings), float(computer_cost), float(potential_income_increase)

def affordability_check(monthly_income, monthly_expenses, monthly_debts, current_savings, computer_cost, potential_income_increase):

    # Convert string input to floats
    monthly_income = float(monthly_income)
    monthly_expenses = float(monthly_expenses)
    monthly_debts = float(monthly_debts)
    current_savings = float(current_savings)
    computer_cost = float(computer_cost)
    potential_income_increase = float(potential_income_increase)

    # Calculate affordability metrics
    if any(value < 0 for value in [monthly_income, monthly_expenses, monthly_debts, current_savings, computer_cost, potential_income_increase]):
                print("All values should be positive. Please try again.")
                return

            # Calculate affordability metrics
    disposable_income = monthly_income - monthly_expenses - monthly_debts
    if disposable_income + potential_income_increase == 0:
        print("Your disposable income plus potential income increase cannot be zero.")
        return

    months_to_save = computer_cost / (disposable_income + potential_income_increase)
    savings_ratio = current_savings / computer_cost

    # Decision criteria
    if disposable_income <= 0:
        decision = "You cannot afford the computer right now due to a lack of disposable income."
    elif savings_ratio >= 1:
        decision = "You can afford the computer right now with your current savings."
    elif months_to_save <= 12:
        decision = "You can save for the computer within a year."
    else:
        decision = "It may take more than a year to save for the computer."

    # Print the results
    print(f"Disposable income: ${disposable_income:.2f}")
    print(f"Months to save for the computer: {months_to_save:.2f}")
    print(f"Savings ratio (savings/computer cost): {savings_ratio:.2f}")
    print(decision)
    input("\nPress enter to continue...")

    # Example usage
    metrics, decision = affordability_metrics(
       monthly_income,
       monthly_expenses,
       monthly_debts,
       current_savings,
       computer_cost,
       potential_income_increase
    )

    print(f"Metrics: {metrics}")
    print(f"Decision: {decision}")

    ai_decision = questionary.confirm("Would you like to see the AI's decision?").ask()
    if ai_decision:
        print("")
        openai_api_key = settings['openai_api_key']
        expenses = ", ".join([f"{item}: ${cost:.2f}" for item, cost in getExpenses()])
        response = purchase_suggest_gpt(expenses, computer_cost)
        print(f"AI Decision: {response}")
        
    keep_going = questionary.confirm("Would you like to go back to the menu?").ask()
    if keep_going:
        menu()



def menu():
    """
    Displays an interactive menu to manage expenses.
    """
    # Import ascii title library
    from pyfiglet import Figlet
    f = Figlet(font='slant')
    global settings
    settings = yaml.safe_load(open("settings.yaml"))
    
    while True:
        import os
        os.system('clear') if os.name == 'posix' else os.system('cls')
        print(f.renderText('PyBudget'))
        print('[bold cyan]========= MENU =========[/bold cyan]')
        choice = questionary.select(
            "Please select an option:",
            choices=[
                "Add an expense",
                "Delete expense",
                "View expenses",
                "Optimize budget",
                "Plan a purchase",
                "Check statistics",
                "Settings",
                "Exit"
            ]).ask()

        if choice == "Add an expense":
            name = questionary.text("Enter the name of the expense:").ask()
            amount = questionary.text(
                "Enter the amount of the expense:",
                validate=lambda text: text.replace('.', '', 1).isdigit()
            ).ask()
            amount = getDecimal(float(amount))
            addExpense(f'{name},{amount}')
            input("\nPress enter to continue...")
        elif choice == "Delete expense":
            name = questionary.text("Enter the name of the expense to delete:").ask()
            deleteExpense(name)
            input("\nPress enter to continue...")
        elif choice == "View expenses":
            print("\n[bold red]========= EXPENSES =========[/bold red]")
            itemCosts = getExpenses()
            for item, cost in itemCosts:
                text = f'{item}: ${cost:.2f}'
                info(text)
            input("\nPress enter to continue...")
        elif choice == "Check statistics":
            items = getExpenses()
            display_statistics(items)
        elif choice == "Plan a purchase":
            monthly_income, monthly_expenses, monthly_debts, current_savings, computer_cost, potential_income_increase = ask_for_financial_metrics()
            metrics, decision, detailed_report = detailed_affordability_metrics(monthly_income, monthly_expenses, monthly_debts, current_savings, computer_cost, potential_income_increase)
            print(f"\n[bold magenta]========= AFFORDABILITY METRICS =========[/bold magenta]")
            info(f"Debt-to-Income Ratio: {metrics['dti_ratio']:.2f}%")
            info(f"Savings Ratio: {metrics['savings_ratio']:.2f}%")
            info(f"Expense-to-Income Ratio: {metrics['expense_ratio']:.2f}%")
            info(f"Liquidity Ratio: {metrics['liquidity_ratio']:.2f}")
            info(f"Return on Investment: {metrics['roi']:.2f}") if potential_income_increase else None
            info(f"Financial Flexibility Ratio: {metrics['flexibility_ratio']:.2f}%")
            info(f"Interest Paid on Debt: ${metrics['interest_paid_on_debt']:.2f}") if monthly_debts else None
            info(f"Total Cost with Financing: ${metrics['total_cost_with_financing']:.2f}") if metrics['total_cost_with_financing'] != computer_cost else None
            info(f"Depreciated ROI: {metrics['depreciated_roi']:.2f}") if potential_income_increase else None
            info(f"Break Even Months: {metrics['break_even_months']:.2f}") if potential_income_increase else None
            info(f"Ideal Savings Respected: {metrics['ideal_savings_respected']:.2f}%")
            print(f"\n[bold magenta]========= DECISION =========[/bold magenta]")
            info(f"Decision: {decision}")
            print("")
            paragraph(detailed_report)
            
            print("")
            ai_decision = questionary.confirm("Would you like to see the AI's decision?").ask()
            if ai_decision:
                openai_api_key = settings['openai_api_key']
                expenses = ", ".join([f"{item}: ${cost:.2f}" for item, cost in getExpenses()])
                response = purchase_suggest_gpt(expenses, computer_cost)
                paragraph(f"AI Decision: {response}")
            input("\nPress enter to continue...")
        elif choice == "Optimize budget":
            expenses = ", ".join([f"{item}: ${cost:.2f}" for item, cost in getExpenses()])
            response = optimize_budget_gpt(expenses)
            print("")
            print(f"\n[bold magenta]========= OPTIMIZED BUDGET =========[/bold magenta]")
            paragraph(f"Suggestion: {response}")
            input("\nPress enter to continue...")
        elif choice == "Settings":
            print("\n[bold red]========= SETTINGS =========[/bold red]")
            print(f"Name: {settings['name']}")
            print(f"Age: {settings['age']}")
            print(f"Monthly Income: ${settings['monthly_income']}")
            print(f"Monthly Rent: ${settings['monthly_rent']}")
            print(f"Current Savings: ${settings['current_savings']}")
            print(f"Favorite Color: {settings['fav_color']}")
            print(f"OpenAI API Key: {settings['openai_api_key']}")
            print("")
            edit_settings = questionary.confirm("Would you like to edit your settings?").ask()
            if edit_settings:
                settings['name'] = questionary.text("What is your name?").ask()
                settings['age'] = questionary.text("What is your age?").ask()
                settings['monthly_income'] = questionary.text("What is your monthly income?").ask()
                settings['monthly_rent'] = questionary.text("What is your monthly rent?").ask()
                settings['current_savings'] = questionary.text("What is your current savings?").ask()
                settings['fav_color'] = questionary.text("What is your favorite color?").ask()
                settings['openai_api_key'] = questionary.text("What is your OpenAI API key?").ask() if settings['openai_api_key'] == '' else settings['openai_api_key']
                with open("settings.yaml", "w") as file:
                    yaml.dump(settings, file)
                print("Settings updated.")
                input("\nPress enter to continue...")
        elif choice == "Exit":
            break


# ============ ONBOARDING & MAIN ============
if __name__ == '__main__':
    import os
    import yaml
    if os.path.isfile('settings.yaml'):
        menu()
    else:
        import onboarding
        status = False
        while status != True:
            status = onboarding.onboarding()
            if status == False:
                print("Please, try again. Something went wrong with the onboarding process.")
    menu()
