import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from decimal import Decimal
from cohere_ai import generate_prompt
from prompt_templates import construct_prompt

# st.title('Personal Finance Analyzer')


# Create a sidebar for primary tabs
primary_tab = st.sidebar.radio('Select Mode', ['Corporate'])

# Primary Tab: Individuals
if primary_tab == 'Individuals':
    tab1, tab2 = st.tabs(["Enter Data Manually", "Upload Statement (Automatic)"])


    with tab1:

            personal_income = st.number_input("Enter your personal income", min_value=0, step=1, format="%i")
            other_income = st.number_input("Enter other incomes", min_value=0, step=1, format="%i")
        
        
            col1, col2 = st.columns(2)
            
            st.subheader('Monthly Expenses')
            
            st.text("")
            
            with col1:
            
        
                
                st.markdown("#### Fixed Expenses")
                
                
                electricity_bill = st.number_input("Electricity Bill", min_value=0, step=1, format="%i")
                water_bill = st.number_input("Water Bill", min_value=0, step=1, format="%i") 
                phone_bill = st.number_input("Phone Bill", min_value=0, step=1, format="%i")
                food_supplies = st.number_input("Food Supplies", min_value=0, step=1, format="%i")
                school_supplies = st.number_input("School Supplies", min_value=0, step=1, format="%i")
                other_expenses = st.number_input("Other Expenses", min_value=0, step=1, format="%i")
               
            with col2:
                st.markdown("#### Discretionary Expenses")
                grocery_shopping = st.number_input("Grocery shopping", min_value=0, step=1, format="%i")
                gas_refill = st.number_input("Gas refill", min_value=0, step=1, format="%i")
                eating_out = st.number_input("Eating out", min_value=0, step=1, format="%i")
                movie = st.number_input("Movie", min_value=0, step=1, format="%i")
                clothing = st.number_input("Clothing", min_value=0, step=1, format="%i")
        
            st.subheader('Investment Savings')
            savings = st.number_input("Enter your savings", min_value=0, step=1, format="%i")
            investments = st.number_input("Enter your Investment", min_value=0, step=1, format="%i")
            
            if (personal_income > 0) and (electricity_bill > 0 or water_bill > 0 or phone_bill > 0 or food_supplies > 0 or school_supplies > 0) and (savings > 0):
    
                total_income = personal_income + other_income
                needs = electricity_bill + water_bill + food_supplies + school_supplies + phone_bill + other_expenses 
                wants = grocery_shopping + gas_refill + eating_out + movie + clothing
                investments_savings = savings + investments
                total_expenses = needs + wants + investments_savings
        
                leftover_income = total_income - total_expenses
        
                if leftover_income < 0:
                    st.error("Your expenses exceed your income!")
                else:
                    st.markdown(f"**Leftover income:** SAR {leftover_income}")
    
    
                    # Rule-based report         
                    needs_percentage = needs/total_income  
                    wants_percentage = wants/total_income
                    savings_percentage = (investments_savings + leftover_income) /total_income

        
                    labels = ['Needs', 'Wants', 'Savings']
                    sizes = [needs_percentage, wants_percentage, savings_percentage]
                    fig1, ax1 = plt.subplots()
                    ax1.pie(sizes, labels=labels, autopct='%1.1f%%')
                    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        
                    st.pyplot(fig1)
                    st.markdown("---")            
        
        
                    st.subheader("Report")
                    
                    needs_margin = 0.05  # 10% margin for needs
                    wants_margin = 0.05  # 10% margin for wants
                    savings_margin = 0.05  # 10% margin for savings
                    
                    
                def view_quick_analysis():
                    st.markdown("")
                    try:
                        if needs_percentage > (0.5):
                            st.markdown("#### Needs Expenses:")
                            st.write(f"Heads up! According to the 50-30-20 rule, only 50% of your income should be allocated to needs. Currently, you are overshooting this mark by more than {needs_percentage:.2%}. Consider revisiting your necessities to see if any adjustments can be made.")
                        elif needs_percentage < (0.5):
                            st.markdown("#### Needs Expenses:")
                            st.write(f"Impressive! Your needs expenses are significantly below the 50% target outlined by the 50-30-20 rule. This efficient budgeting on necessities offers more flexibility for your wants or savings.")
                
                        if wants_percentage > (0.3):
                            st.markdown("#### Wants Expenses:")
                            st.write(f"Caution! Your discretionary spending has exceeded the 30% mark set by the 50-30-20 rule by more than {wants_percentage:.2%}. It might be a good idea to scrutinize these expenses to see where you can potentially cut back.")
                        elif wants_percentage < (0.3):
                            st.markdown("#### Wants Expenses:")
                            st.write(f"Well done! You are maintaining your discretionary spending well below the 30% guideline from the 50-30-20 rule. This disciplined management could be your secret to financial comfort.")
                
                        if savings_percentage < (0.2):
                            st.markdown("#### Savings and Investments:")
                            st.write(f"Just a reminder, the 50-30-20 rule encourages allocating 20% of your income to savings. Currently, your savings fall more than {savings_percentage:.2%} below this target. If circumstances permit, try to boost your savings.")
                        elif savings_percentage > (0.2):
                            st.markdown("#### Savings and Investments:")
                            st.write(f"Exceptional! You are setting a great example by saving much more than the 20% target set by the 50-30-20 rule. Keep this up and you are on your way to achieving substantial financial security.")
                    except NameError:
                        st.warning("It appears your expenses may be exceeding your income. Please review your inputs and make sure all required fields are entered correctly.")
                    


                def view_smart_analysis(prompt):
                    analysis = generate_prompt(prompt)
                    st.write((analysis))
                    
                
                template = construct_prompt(total_income, needs, wants, leftover_income, electricity_bill, water_bill, phone_bill,
                          food_supplies, school_supplies, other_expenses, grocery_shopping, gas_refill, eating_out,
                          movie, clothing)
                
                
                
               # Create four columns
                col1, col2, col3= st.columns([2,1,1])
                
                # Use the second and third columns to create the selectbox and button
                with col1:
                    option = st.selectbox('Please choose an analysis type', ('Quick Analysis', 'Smart Analysis'))
                    
                    if st.button('View Report'):
                        if option == 'Quick Analysis':
                            view_quick_analysis()
                        elif option == 'Smart Analysis':
                            view_smart_analysis(prompt=template)
                            



    with tab2:
    
  
        basic_expenses = ["Rent", "Utilities", "Electric bill", "Water bill", "Internet bill",
                      "Phone bill", "Car maintenance", "Loan repayment", "Insurance"]
    
        variable_expenses = ["Grocery shopping", "Gas refill", "Eating out", "Movie", "Clothing",
                          "Electronics", "Entertainment", "Transportation", "Education", "Gifts",
                          "Charity", "Vacation", "Fitness", "Books", "Pets", "Home improvement",
                          "Beauty", "Household goods", "Public transport", "Healthcare"]
    

        
        uploaded_file = st.file_uploader("Upload your bank account statement", type='csv')
    
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            data = data.sort_values('Date')  # Ensure data is sorted by date
    
            # CHARTS SECTION
            st.markdown("## Charts Section")
    
            # Remove rows where Category is 'Salary' for the charts
            
            def plot_chart(chart_type):
                chart_data = data[data['Details'] != 'Salary']
        
                grouped_data = chart_data.groupby('Details', as_index=False)['Debit'].sum()
                total_spent = grouped_data['Debit'].sum()
        
                if chart_type == 'bar':
                    fig_bar = px.bar(grouped_data, y='Debit', x='Details', text='Debit')
                    fig_bar.update_traces(texttemplate='%{text:.2s}', textposition='outside')
                    fig_bar.update_layout(uniformtext_minsize=14, uniformtext_mode='hide',
                                      autosize=True,
                                      margin=dict(l=0, r=0, b=0, t=0, pad=0),
                                      xaxis=dict(
                                        title='Category',
                                        titlefont_size=16,
                                        tickfont_size=14,
                                    )                       
                                      )
                    st.plotly_chart(fig_bar)
                    
                elif chart_type == 'pie':
                    fig_pie = px.pie(grouped_data, values='Debit', names='Details')
                    st.plotly_chart(fig_pie)
            
            
            import plotly.graph_objects as go
            needs = sum([data[data['Details'] == i].sum()['Debit'] for i in basic_expenses])
            wants = sum([data[data['Details'] == i].sum()['Debit'] for i in variable_expenses])
            # investments_savings =  # data[(data['Details'] == "Savings") | (data['Details'] == "Investments")].sum()['Credit']
            total_expenses = needs + wants
            total_income = data['Credit'].sum()
            leftover_income = total_income - total_expenses
            
            if leftover_income < 0:
                st.error("Your expenses exceed your income!")
            
            # Rule-based report
            needs_percentage = needs/total_income
            wants_percentage = wants/total_income
            savings_percentage = leftover_income / total_income
            
            def plot_pie_chart_50_30_20():
    
                
                labels = ['Needs', 'Wants', 'Savings']
                values = [needs_percentage, wants_percentage, savings_percentage]
                
                fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3, textinfo='percent+label',
                                              texttemplate='%{percent:.1%}')])
                
                fig.update_traces(hoverinfo='label+percent', textfont_size=20,
                                  marker=dict(colors=['#FDD470', '#3498db', '#2ecc71'], 
                                              line=dict(color='#FFFFFF', width=2)))
                
                fig.update_layout(
                title_text="Expense Distribution",
                annotations=[dict(text='Expenses', x=0.5, y=0.5, font_size=20, showarrow=False)])
                
                st.plotly_chart(fig)
            
            
            def plot():
                from plotly.subplots import make_subplots
                # Pivot the data (create new DataFrame)
                df_pivot = data.pivot_table(values='Debit', index='Date', columns='Details', aggfunc='sum', fill_value=0)
                
                # Create subplot with secondary y-axis
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Calculate width of bars in milliseconds (1 day)
                bar_width = 24*60*60*1000  # 1 day in milliseconds
                
                # Add Stacked Bar Plot for each category
                for category in df_pivot.columns:
                    fig.add_trace(go.Bar(x=df_pivot.index, y=df_pivot[category], name=category, width=bar_width), secondary_y=False)
                
                # Add Line Plot for Balance
                fig.add_trace(go.Scatter(x=data['Date'], y=data['Balance'], mode='lines', name='Balance', line=dict(color='blue')), secondary_y=True)
                
                # Add Line at Balance = 0
                fig.add_trace(go.Scatter(x=data['Date'], y=[0]*len(data), mode='lines', name='Zero Balance', 
                                          line=dict(color='red', dash='dash')), secondary_y=True)
                
                # Update yaxis titles
                fig.update_layout(title='Spending Categories and Balance Over Time',
                                  yaxis_title='Total Debit',
                                  yaxis2_title='Balance')
                
                st.plotly_chart(fig)
    
            option = st.selectbox(
                'How would you like to display the graph?',
                ('Pie Chart (Each Category)', 'Pie Chart (50-30-20)', 'Bar Graph', 'Bar Graph + Balance'))
            
            if option == 'Pie Chart (Each Category)':
                plot_chart('pie')
            elif option == 'Pie Chart (50-30-20)':
                plot_pie_chart_50_30_20()
            elif  option == 'Bar Graph':
                plot_chart('bar')
            elif option == 'Bar Graph + Balance':
                plot()
                
    
    
    
            # SUMMARY TABLE SECTION
            st.markdown("## Summary Table Section")
    
            initial_balance = data.iloc[0]['Balance']  # The balance of the first entry
            closing_balance = data.iloc[-1]['Balance']  # The balance of the last entry
    
            deposits = data[data['Credit'] > 0]
            num_deposits = len(deposits)  # Number of deposits
            total_deposit_amount = deposits['Credit'].sum()  # Total deposit amount
    
            deductions = data[data['Debit'] > 0]  # Filter for deductions (Debit) excluding zero values
            num_deductions = len(deductions)  # Number of deductions
            total_amount_deducted = deductions['Debit'].sum()  # Total amount deducted
    
            salary = data[data['Details'] == 'Salary']['Credit'].sum()
            max_expenses = data.groupby('Details')['Debit'].sum().idxmax()
            min_expenses = data.groupby('Details')['Debit'].sum().loc[lambda x : x > 0].idxmin()
            
            # Create a dataframe to display the results
            df_summary = pd.DataFrame({
                'Metric': ['Initial Balance', 'Closing Balance', 'Number of Deposits', 'Total Deposit Amount',
                    'Number of Deductions', 'Total Amount Deducted', 'Salary', 'Maximum Spending', 'Minimum Spending',
                    'leftover income'],
                'Value': [initial_balance,closing_balance, num_deposits, total_deposit_amount, 
                          num_deductions, total_amount_deducted, salary, max_expenses, 
                          min_expenses, leftover_income]
    
            })
    
            def remove_index():      
                # CSS to inject contained in a string
                hide_table_row_index = """
                            <style>
                            thead tr th:first-child {display:none}
                            tbody th {display:none}
                            </style>
                            """
                # Inject CSS with Markdown
                st.markdown(hide_table_row_index, unsafe_allow_html=True)
                
    
    
            # remove_index()
            st.table(df_summary)
        

# Primary Tab: Companies
elif primary_tab == 'Corporate':

    # Create sub-tab for Companies
    tab1, tab2, tab3, tab4 = st.tabs(["Income Statement", "Income Statement (Automatic)", "Balance Sheet", "Balance Sheet (Automatic)"])
    
    # option = st.selectbox(
    #     "What type would you like to use?", 
    #     ('Income Statement', 'Statement of financial position'))
    
    # Display relevant content for sub-tab under Companies
    with tab1: 

        st.header("Financial Analysis")
        col1, col2 = st.columns(2)
        
        
        with col1:
            # Define a simple dictionary to store our data.
            data = {
                'Revenue Header': st.markdown("#### Revenue"),
                'Net Sales': st.number_input('Enter Net Sales', min_value=0),
                'Other Income': st.number_input('Enter Other Income', min_value=0),
                
                'Expense Header': st.markdown("#### Expense"),
                'Cost of Goods Sold': st.number_input('Enter Cost of Goods Sold', min_value=0, format="%i"),
                'Selling and Operating Expenses': st.number_input('Enter Selling and Operating Expenses', min_value=0, format="%i", value=0),
                'General and Administration Expenses': st.number_input('Enter General and Administration Expenses', min_value=0, format="%i", value=0),
                'Interest Expense': st.number_input('Enter Interest Expense', value=0),
                'Other Expense': st.number_input('Enter Other Expense', min_value=0),
                
                'Other Header': st.markdown("#### Other"),
                'Gain/Loss on Financial Instruments': st.number_input('Enter Gain/Loss on Financial Instruments', step=1, format="%i", value=0),
                'Gain/Loss on Foreign Currency': st.number_input('Enter Gain/Loss on Foreign Currency', min_value=-10000000, format="%i", value=0),
                
                'Tax Header': st.markdown("#### Tax"),
                'Income Tax Expense': st.number_input('Enter Income Tax Expense', min_value=0, format="%i", value=0)
            }

        
            # Calculate the different components of the income statement
            gross_profit = data['Net Sales'] - data['Cost of Goods Sold']
            total_operating_expense = data['General and Administration Expenses'] + data['Selling and Operating Expenses']
            operating_expense = gross_profit - total_operating_expense
            non_operating_expense = data['Gain/Loss on Financial Instruments']  + data['Gain/Loss on Foreign Currency'] + data['Other Income'] + data['Other Expense'] + data['Interest Expense'] 
            EBIT = gross_profit - total_operating_expense + non_operating_expense
            net_income = EBIT - data['Income Tax Expense']

        # Calculate the ratios
            if data['Net Sales'] != 0:
                gross_margin = (gross_profit / data['Net Sales']) * 100
                profit_margin = (net_income / data['Net Sales']) * 100
                operating_margin = (total_operating_expense / data['Net Sales']) * 100
            else:
                gross_margin = 0
                profit_margin = 0
                operating_margin = 0
            
        with col2:
            
            st.subheader('Income Statment:')
            # Display the results
            st.markdown(f'Net Sales: {format(data["Net Sales"], ",")}')
            st.markdown(f'Cost of Goods Sold: {format(data["Cost of Goods Sold"], ",")}')
            st.markdown(f'&nbsp;&nbsp;&nbsp;&nbsp; **Gross Profit:** {format(gross_profit, ",")}')
            st.markdown('')
            st.markdown(f'Selling and Operating Expenses: {format(data["Selling and Operating Expenses"], ",")}')
            st.markdown(f'General and Administration Expenses: {format(data["General and Administration Expenses"], ",")}')  
            st.markdown(f'&nbsp;&nbsp;&nbsp;&nbsp; **Total Operating Expense:** {format(total_operating_expense, ",")}')
            st.markdown(f'&nbsp;&nbsp;&nbsp;&nbsp; **Operating Income:** {format(operating_expense, ",")}')
            # st.markdown(f'&nbsp;&nbsp;&nbsp;&nbsp; **Non Operating Expense:** {format(non_operating_expense, ",")}')
            st.markdown('')
            st.markdown(f'Other Income: {format(data["Other Income"], ",")}')
            st.markdown(f'Gain/Loss on Financial Instruments: {format(data["Gain/Loss on Financial Instruments"], ",")}')
            st.markdown(f'Gain/Loss on Foreign Currency: {format(data["Gain/Loss on Foreign Currency"], ",")}')
            st.markdown(f'Interest Expense: {format(data["Interest Expense"], ",")}')
            st.markdown(f'Other Expense: {format(data["Other Expense"], ",")}')
            st.markdown('')
            st.markdown(f'**Total Income Before Tax (EBIT):** {format(EBIT, ",")}')
            st.markdown(f'&nbsp;&nbsp;&nbsp;&nbsp;Income Tax Expense: {format(data["Income Tax Expense"], ",")}')
            st.markdown(f'**Net Income:** {format(net_income, ",")}')
            
            st.markdown('')
            st.markdown("""<hr style='border:1px solid blue'> """, unsafe_allow_html=True)
            
            st.markdown(f'**Gross Profit Margin:** The company keeps {gross_margin:.2f}% for every Riyal it makes and the remainer goes on operating expenses')
            st.markdown(f'**Operation Margin:** The company keeps {operating_margin:.2f}% revenue is left over after payign variable costs')
            st.markdown(f'**Net Margin:** Net income was {profit_margin:.2f}% for every Riyal generated')
            

        # First, we define a list with the labels (in this case, the names of the different metrics)
        labels = ['Net Sales', 'Gross Profit', 'Operating Expense', 'Net Income']
        
        # Then, we create a list with the corresponding values
        values = [data["Net Sales"], gross_profit, total_operating_expense, net_income]
        values = [format(i, ",") for i in values]
        
        # Define a list of colors for each bar
        colors = ['blue', 'green', 'red', 'orange']
        # Create the bar chart
        fig = go.Figure(data=[go.Bar(
            
            x=labels,
            y=values,
            text=values,
            textposition='auto',
            marker_color=colors,
            
        )])
        
        # Add title and labels
        fig.update_layout(
            title_text='Financial Metrics',
            xaxis_title="Metrics",
            yaxis_title="Amount",
        )
        
        # Show the plot
        st.plotly_chart(fig)
        
    with tab2:

        uploaded_file = st.file_uploader("Upload your income statement", type='csv')

        if uploaded_file is not None:
            # Read the file into a DataFrame
            data = pd.read_csv(uploaded_file)
            # Check if the DataFrame has the expected columns
            if 'Description' in data.columns and 'Amount' in data.columns and len(data) == 10:

                # Convert DataFrame to a dictionary
                data = data.set_index('Description').T.to_dict('records')[0]

                # Calculate the different components of the income statement
                gross_profit = data['Net Sales'] - data['Cost of goods sold']
                total_operating_expense = data['General and Administration Expenses'] + data['Selling and Operating Expenses']
                operating_expense = gross_profit - total_operating_expense
                non_operating_expense = data['Gain/Loss on Financial Instruments'] + data['Gain/Loss on Foreign Currency'] + \
                                        data['Other Income'] + data['Other Expense'] + data['Interest Expense']
                EBIT = gross_profit - total_operating_expense + non_operating_expense
                net_income = EBIT - data['Income Tax Expense']

                # Calculate the ratios
                if data['Net Sales'] != 0:
                    gross_margin = (gross_profit / data['Net Sales']) * 100
                    profit_margin = (net_income / data['Net Sales']) * 100
                    operating_margin = (total_operating_expense / data['Net Sales']) * 100
                else:
                    gross_margin = 0
                    profit_margin = 0
                    operating_margin = 0

                st.subheader('Income Statment:')
                # Display the results
                st.markdown(f'Net Sales: {format(data["Net Sales"], ",")}')
                st.markdown(f'Cost of Goods Sold: {format(data["Cost of goods sold"], ",")}')
                st.markdown(f'&nbsp;&nbsp;&nbsp;&nbsp; **Gross Profit:** {format(gross_profit, ",")}')
                st.markdown('')
                st.markdown(f'General and Administration Expenses: {format(data["General and Administration Expenses"], ",")}')
                st.markdown(f'Selling and Operating Expenses: {format(data["Selling and Operating Expenses"], ",")}')
                st.markdown(f'&nbsp;&nbsp;&nbsp;&nbsp; **Total Operating Expense:** {format(total_operating_expense, ",")}')
                st.markdown(f'&nbsp;&nbsp;&nbsp;&nbsp; **Operating Income:** {format(operating_expense, ",")}')
                # st.markdown(f'&nbsp;&nbsp;&nbsp;&nbsp; **Non Operating Expense:** {format(non_operating_expense, ",")}')
                st.markdown('')
                st.markdown(f'Other Income: {format(data["Other Income"], ",")}')
                st.markdown(f'Gain/Loss on Financial Instruments: {format(data["Gain/Loss on Financial Instruments"], ",")}')
                st.markdown(f'Gain/Loss on Foreign Currency: {format(data["Gain/Loss on Foreign Currency"], ",")}')
                st.markdown(f'Interest Expense: {format(data["Interest Expense"], ",")}')
                st.markdown(f'Other Expense: {format(data["Other Expense"], ",")}')
                st.markdown('')
                st.markdown(f'**Total Income Before Tax (EBIT):** {format(EBIT, ",")}')
                st.markdown(f'&nbsp;&nbsp;&nbsp;&nbsp;Income Tax Expense: {format(data["Income Tax Expense"], ",")}')
                st.markdown(f'**Net Income:** {format(net_income, ",")}')
                st.markdown('')
                st.markdown("""<hr style='border:1px solid blue'> """, unsafe_allow_html=True)
                st.markdown(f'**Gross Profit Margin:** The company keeps {gross_margin:.2f}% for every Riyal it makes and the remainer goes on operating expenses')
                st.markdown(f'**Operation Margin:** The company keeps {operating_margin:.2f}% revenue is left over after payign variable costs')
                st.markdown(f'**Net Margin:** Net income was {profit_margin:.2f}% for every Riyal generated')

                # First, we define a list with the labels (in this case, the names of the different metrics)
                labels = ['Net Sales', 'Gross Profit', 'Operating Expense', 'Net Income']

                # Then, we create a list with the corresponding values
                values = [data["Net Sales"], gross_profit, total_operating_expense, net_income]
                values = [format(i, ",") for i in values]

                # Define a list of colors for each bar
                colors = ['blue', 'green', 'red', 'orange']
                # Create the bar chart
                fig = go.Figure(data=[go.Bar(

                    x=labels,
                    y=values,
                    text=values,
                    textposition='auto',
                    marker_color=colors,

                )])

                # Add title and labels
                fig.update_layout(
                    title_text='Financial Metrics',
                    xaxis_title="Metrics",
                    yaxis_title="Amount",
                )

                # Show the plot
                st.plotly_chart(fig)

            else:
                st.error("Invalid file format. Please upload a CSV file with 'Description' and 'Amount' columns.")

    with tab3:
    
        col1, col2, col3 = st.columns(3)
    
        with col1:
            st.header("Assets")
            st.markdown("#### Current Assets")
            cash_equivalents = st.number_input("Cash and cash equivalents", min_value=0.0, format="%f")
            accounts_receivable = st.number_input("Accounts receivable and prepayments", min_value=0.0, format="%f")
            inventory = st.number_input("Inventory", min_value=0.0, format="%f")
            other_short_term_assets = st.number_input("Other short term assets", min_value=0.0, format="%f")
    
            st.markdown("#### Non-current Assets")
            ppe = st.number_input("Property, Plant & Equipment (PPE)", min_value=0.0, format="%f")
            long_term_investments = st.number_input("Long-term investments", min_value=0.0, format="%f")
            intangible_assets = st.number_input("Intangible assets", min_value=0.0, format="%f")
            deferred_charges = st.number_input("Deferred charges and other noncurrent assets", format="%f")
    
        with col2:
            st.header("Liabilities and Equity")
            st.markdown("#### Current Liabilities")
            accounts_payable = st.number_input("Accounts payable", min_value=0.0, format="%f")
            short_term_debt = st.number_input("Short term debt", min_value=0.0, format="%f")
            current_portion_long_term_debt = st.number_input("Current portion of long term debt", min_value=0.0, format="%f")
    
            st.markdown("#### Non-current Liabilities")
            long_term_debt = st.number_input("Long-term Debt (loan)", min_value=0.0, format="%f")
    
            st.markdown("#### Equity")
            owners_capital = st.number_input("Owner's Capital", min_value=0.0, format="%f")
            retained_earnings = st.number_input("Retained Earnings", min_value=0.0, format="%f")
            
        with col3:
            st.header("Other")
            opening_balance_ppe = st.number_input("Opining balance PPE", min_value=0.0, format="%f")
            closing_balance_ppe = st.number_input("Closing balance PPE", value=ppe, format="%f")
            opening_balance_inventory = st.number_input("Opining balance Inventory", min_value=0.0, format="%f")
            closing_balance_inventory = st.number_input("Closing balance Inventory", value=inventory, format="%f")
            opening_accounts_receivable = st.number_input("Opining Accounts Receivable", min_value=0.0, format="%f")
            closing_accounts_receivable = st.number_input("Closing Accounts Receivable", value=accounts_receivable, format="%f")
            net_sales = st.number_input("Net Sales", min_value=0.0, format="%f")
            cost_of_goods_sold = st.number_input("Cost of Goods Sold", min_value=0.0, format="%f")
            net_income = st.number_input("Net Income", value=(net_sales-cost_of_goods_sold), format="%f")
            

    
        # Perform operations
        total_current_assets = cash_equivalents + accounts_receivable + inventory + other_short_term_assets
        total_non_current_assets = ppe + long_term_investments + intangible_assets + deferred_charges
        total_assets = total_current_assets + total_non_current_assets
    
        total_current_liabilities = accounts_payable + short_term_debt + current_portion_long_term_debt
        total_non_current_liabilities = long_term_debt
        total_liabilities = total_current_liabilities + total_non_current_liabilities
    
        total_equity = owners_capital + retained_earnings
        total_liabilities_and_equity = total_liabilities + total_equity
        
        average_ppe = (closing_balance_ppe + opening_balance_ppe) / 2
        average_inventory = (closing_balance_inventory + opening_balance_inventory) / 2
        average_accounts_receivable = (closing_accounts_receivable + opening_accounts_receivable) / 2
        
        try: fixed_assets_turnover_ratio = net_sales/average_ppe
        except ZeroDivisionError: fixed_assets_turnover_ratio = 'N/A (Cannot divide by zero)'
            
        try: current_ratio = total_current_assets/total_current_liabilities
        except ZeroDivisionError: current_ratio = 'N/A (Cannot divide by zero)'
        
        quick_assets = cash_equivalents + accounts_receivable
        
        try: quick_ratio = quick_assets / total_current_liabilities
        except: quick_ratio = 'N/A (Cannot divide by zero)'
            
        try: inventory_turnover_ratio = cost_of_goods_sold / average_inventory
        except: inventory_turnover_ratio = 'N/A (Cannot divide by zero)'
        
        try: inventory_turnover_ratio_by_day = 365 / inventory_turnover_ratio
        except: inventory_turnover_ratio_by_day = 'N/A (Cannot divide by zero)'
        
        try: accounts_receivable_turnover_ratio = net_sales / average_accounts_receivable
        except: accounts_receivable_turnover_ratio = 'N/A (Cannot divide by zero)'
        
        try: debt_to_equity_ratio = long_term_debt / total_equity
        except: debt_to_equity_ratio = 'N/A (Cannot divide by zero)'
        
        try: return_on_equity = net_income / total_equity 
        except: return_on_equity = 'N/A (Cannot divide by zero)'
        
        try: debt_equity_ratio = total_liabilities / total_equity
        except: debt_equity_ratio = 'N/A (Cannot divide by zero)'
            

        
        # expander_view_report = st.expander("View report")
        # expander_view_analysis = st.expander("View Analysis")
        # expander_view_smart_analysis = st.expander("View Smart Analysis")

        # Use the expander instead of st when you want to put something inside the expandable box
        def view_report():
            
            # Display the calculated results
            
            st.markdown("#### Assets")
            st.markdown(f'**Current Assets:**')
            st.markdown(f'&nbsp;&nbsp;&nbsp;&nbsp;Cash and Cash Equivalents: {cash_equivalents:,.2f}')
            st.markdown(f'&nbsp;&nbsp;&nbsp;&nbsp;Accounts Receivable and Prepayments: {accounts_receivable:,.2f}')
            st.markdown(f'&nbsp;&nbsp;&nbsp;&nbsp;Inventory: {inventory:,.2f}')
            st.markdown(f'&nbsp;&nbsp;&nbsp;&nbsp;Other Short Term Assets: {other_short_term_assets:,.2f}')
            st.markdown(f'**Total Current Assets:** {total_current_assets:,.2f}')
            st.markdown("""<hr style='border:1px solid blue'> """, unsafe_allow_html=True)
            
            
            st.markdown(f'**Non Current Assets:**')
            st.markdown(f'&nbsp;&nbsp;&nbsp;&nbsp;Property, Plant & Equipment (PPE): {ppe:,.2f}')
            st.markdown(f'&nbsp;&nbsp;&nbsp;&nbsp;Long-term Investments: {long_term_investments:,.2f}')
            st.markdown(f'&nbsp;&nbsp;&nbsp;&nbsp;Intangible Assets: {intangible_assets:,.2f}')
            st.markdown(f'&nbsp;&nbsp;&nbsp;&nbsp;Deferred Charges and Other Noncurrent Assets: {deferred_charges:,.2f}')
            st.markdown(f'**Total Non-current Assets:** {total_non_current_assets:,.2f}')
            st.markdown("""<hr style='border:1px solid blue'> """, unsafe_allow_html=True)
            
            st.markdown(f'**Total Assets:** {total_assets:,.2f}')
            st.markdown("""<hr style='border:1px solid blue'> """, unsafe_allow_html=True)
            
            st.markdown("#### Liabilities and Equity")
            st.markdown(f'**Liabilities:**')
            st.markdown(f'**Current Liabilities:**')
            
            st.markdown(f'&nbsp;&nbsp;&nbsp;&nbsp;Accounts Payable: {accounts_payable:,.2f}')
            st.markdown(f'&nbsp;&nbsp;&nbsp;&nbsp;Short Term Debt: {short_term_debt:,.2f}')
            st.markdown(f'&nbsp;&nbsp;&nbsp;&nbsp;Current Portion of Long Term Debt: {current_portion_long_term_debt:,.2f}')
            st.markdown(f'**Total Current Liabilities:** {total_current_liabilities:,.2f}')
            st.markdown("""<hr style='border:1px solid blue'> """, unsafe_allow_html=True)
            
            st.markdown(f'**Noncurrent Liabilities:**')
            st.markdown(f'&nbsp;&nbsp;&nbsp;&nbsp;Long-term Debt (loan): {long_term_debt:,.2f}')
            st.markdown(f'**Total Non-current Liabilities:** {total_non_current_liabilities:,.2f}')
            st.markdown(f'**Total Liabilities:** {total_liabilities:,.2f}')
            st.markdown("""<hr style='border:1px solid blue'> """, unsafe_allow_html=True)
            
            st.markdown(f'**Equity:**')
            st.markdown(f'&nbsp;&nbsp;&nbsp;&nbsp;Owner\'s Capital: {owners_capital:,.2f}')
            st.markdown(f'&nbsp;&nbsp;&nbsp;&nbsp;Retained Earnings:{retained_earnings:,.2f}')
            st.markdown(f'**Total Equity:** {total_equity:,.2f}')
            st.markdown("<hr style='border:1px solid blue'> ", unsafe_allow_html=True)
            st.markdown(f'**Total Liabilities & Equity:** {total_liabilities_and_equity:,.2f}')
            
            st.markdown("<hr style='border:1px solid blue'> ", unsafe_allow_html=True)
            
            st.markdown(f'**Average PPE:** {average_ppe:,.2f}')
            st.markdown(f'**Average Inventory:** {average_inventory:,.2f}')
            st.markdown(f'**Average Accounts Receivable:** {average_accounts_receivable:,.2f}')
            
            
        
            def display_ratio(ratio_name, ratio_value):
                if isinstance(ratio_value, str):
                    st.markdown(f'<p style="color:red;">**Error calculating {ratio_name}:** {ratio_value}</p>', unsafe_allow_html=True)
                else:
                    st.markdown(f'**{ratio_name}:** {ratio_value:,.2f}')
    
            # Create a dictionary mapping ratio names to their calculated values
            ratios = {
                'Fixed Assets Turnover Ratio': fixed_assets_turnover_ratio,
                'Quick Assets': quick_assets,
                'Current Ratio': current_ratio,
                'Inventory Turnover Ratio': inventory_turnover_ratio,
                'Inventory Turnover Ratio by Day': inventory_turnover_ratio_by_day,
                'Accounts Receivable Turnover Ratio': accounts_receivable_turnover_ratio,
                'Debt to Equity Ratio': debt_to_equity_ratio,
                'Return on Equity (ROE)': return_on_equity,
                'Debt Equity Ratio': debt_equity_ratio,
            }

            # Loop through the dictionary and display each ratio
            for ratio_name, ratio_value in ratios.items():
                display_ratio(ratio_name, ratio_value)
                
        def view_quick_analysis_corp():
            if not isinstance(fixed_assets_turnover_ratio, str) and \
                not isinstance(current_ratio, str) and \
                not isinstance(quick_ratio, str) and \
                not isinstance(inventory_turnover_ratio, str) and \
                not isinstance(inventory_turnover_ratio_by_day, str) and \
                not isinstance(accounts_receivable_turnover_ratio, str) and \
                not isinstance(debt_to_equity_ratio, str) and \
                not isinstance(return_on_equity, str) and \
                not isinstance(debt_equity_ratio, str):
        
                st.markdown(f"- For each Riyal invested in fixed assets, the company generates {fixed_assets_turnover_ratio:.2f} in net sales.")
        
                if current_ratio >= 1:
                    st.markdown(f"- With a Current Ratio of {current_ratio:.2f}, the company is well-equipped to meet short-term liabilities with its short-term assets.")
                else:
                    st.markdown(f"- With a Current Ratio of {current_ratio:.2f}, the company may face difficulties in meeting short-term liabilities with its short-term assets.")
        
                if quick_ratio >= 1:
                    st.markdown(f"- A Quick Ratio of {quick_ratio:.2f} indicates the company's solid ability to cover current liabilities with its most liquid assets.")
                else:
                    st.markdown(f"- A Quick Ratio of {quick_ratio:.2f} suggests the company might have issues covering current liabilities with its most liquid assets.")
        
                st.markdown(f"- The company rotates its inventory into finished goods approximately {inventory_turnover_ratio:.2f} times per year, translating to an average of {inventory_turnover_ratio_by_day:.2f} days to convert inventories into saleable products.")
        
                st.markdown(f"- On average, the company collects its receivables {accounts_receivable_turnover_ratio:.2f} times per year.")
        
                st.markdown(f"- The company has {debt_to_equity_ratio:.2f} units of debt for each unit of equity, indicating its leverage position.")
        
                st.markdown(f"- The company generates a profit of {return_on_equity:.2f} per unit of equity, illustrating the return it makes on its own net resources.")
        
                st.markdown(f"- For each unit of equity owned by the company's shareholders, the company owes {debt_equity_ratio:.2f} to its creditors, indicating its debt load relative to its equity.")
                
        
        def view_smart_analysis_corp():
            if not isinstance(fixed_assets_turnover_ratio, str) and \
                not isinstance(current_ratio, str) and \
                not isinstance(quick_ratio, str) and \
                not isinstance(inventory_turnover_ratio, str) and \
                not isinstance(inventory_turnover_ratio_by_day, str) and \
                not isinstance(accounts_receivable_turnover_ratio, str) and \
                not isinstance(debt_to_equity_ratio, str) and \
                not isinstance(return_on_equity, str) and \
                not isinstance(debt_equity_ratio, str):
            
                # Generate a single, comprehensive prompt
                prompt = f"""
                The company's financial performance is as follows:
            
                - Fixed Assets Turnover Ratio: {fixed_assets_turnover_ratio:.2f}
                - Current Ratio: {current_ratio:.2f}
                - Quick Ratio: {quick_ratio:.2f}
                - Inventory Turnover Ratio: {inventory_turnover_ratio:.2f}
                - Inventory Turnover Ratio by Day: {inventory_turnover_ratio_by_day:.2f}
                - Accounts Receivable Turnover Ratio: {accounts_receivable_turnover_ratio:.2f}
                - Debt to Equity Ratio: {debt_to_equity_ratio:.2f}
                - Return on Equity (ROE): {return_on_equity:.2f}
                - Debt Equity Ratio: {debt_equity_ratio:.2f}
            
                Based on this information, what insights and analyses can be drawn?
                """
                # Generate a single AI response
                response = generate_prompt(prompt)
            
                # Display the AI response
                st.markdown(response)
                
        # Create four columns
      
        import time
        # Use the second and third columns to create the selectbox and button
        st.markdown("---")
        st.subheader("Report")
        col1, col2, col3= st.columns([2,1,1])
        with col1:
            option = st.selectbox('Please choose an analysis type', ('Statistics', 'Quick Analysis', 'Smart Analysis'), key='select_box_1')
           
            if st.button('View', key='view_button_1'):
                
                if option == 'Statistics':
                    view_report()
                elif option == 'Quick Analysis':
                    with st.spinner('Wait ...'):
                        time.sleep(1.5)
                        view_quick_analysis_corp()
                        
                    
                elif option == 'Smart Analysis':
                    view_smart_analysis_corp()
                    time.sleep(3)

    with tab4:
        uploaded_file = st.file_uploader("Upload your trial balance", type='csv')

        if uploaded_file is not None:
            # Read the CSV files
            classifications_df = pd.read_csv('utils/classifications.csv')
            data = pd.read_csv(uploaded_file)

            if 'Description' in data.columns and 'Amount' in data.columns and len(data) == 25:

                # Merge the two dataframes based on 'Description'
                merged_df = pd.merge(data, classifications_df, on='Description', how='inner')

                # Group by 'Category' and calculate the sum of 'Amount'
                category_sum = merged_df.groupby('Category')['Amount'].sum()

                # Convert the groupby object to a DataFrame for better visualization
                category_sum_df = category_sum.reset_index()

                # Create variables for specific rows
                accounts_receivable_and_prepayments = category_sum_df.iloc[1, 1]
                cash_and_cash_equivalents = category_sum_df.iloc[2, 1]
                inventory = category_sum_df.iloc[7, 1]
                other_short_term_assets = category_sum_df.iloc[15, 1]
                total_current_assets = accounts_receivable_and_prepayments + cash_and_cash_equivalents + inventory + other_short_term_assets

                property_plant_equipment = category_sum_df.iloc[17, 1]
                long_term_investments = category_sum_df.iloc[9, 1]
                intangible_assets = category_sum_df.iloc[6, 1]
                deferred_charges = category_sum_df.iloc[5, 1]
                total_non_current_assets = property_plant_equipment + long_term_investments + intangible_assets + deferred_charges
                total_assets = total_current_assets + total_non_current_assets

                # Create variables for additional rows
                current_portion_of_long_term_debt = category_sum_df.iloc[4, 1]
                short_term_debt = category_sum_df.iloc[19, 1]
                accounts_payable = category_sum_df.iloc[0, 1]
                total_current_liabilities = current_portion_of_long_term_debt + short_term_debt + accounts_payable

                owner_capital = category_sum_df.iloc[16, 1]
                retained_earnings = category_sum_df.iloc[18, 1]
                total_equity = owner_capital + retained_earnings
                total_liabilities_and_equity = total_liabilities + total_equity

                long_term_debt_loan = category_sum_df.iloc[8, 1]
                total_non_current_liabilities = long_term_debt_loan
                total_liabilities = total_current_liabilities + total_non_current_liabilities

                opening_accounts_receivable = category_sum_df.iloc[12, 1]
                opening_balance_ppe = category_sum_df.iloc[13, 1]
                opening_balance_inventory = category_sum_df.iloc[14, 1]
                closing_accounts_receivable = accounts_receivable_and_prepayments
                closing_balance_ppe = property_plant_equipment
                closing_balance_inventory = inventory
                net_sales = category_sum_df.iloc[11, 1]
                cost_of_goods_sold = category_sum_df.iloc[3, 1]
                net_income = category_sum_df.iloc[10, 1]

                average_ppe = (closing_balance_ppe + opening_balance_ppe) / 2
                average_inventory = (closing_balance_inventory + opening_balance_inventory) / 2
                average_accounts_receivable = (closing_accounts_receivable + opening_accounts_receivable) / 2

                try: fixed_assets_turnover_ratio = net_sales / average_ppe
                except ZeroDivisionError: fixed_assets_turnover_ratio = 'N/A (Cannot divide by zero)'

                try: current_ratio = total_current_assets / total_current_liabilities
                except ZeroDivisionError: current_ratio = 'N/A (Cannot divide by zero)'

                quick_assets = cash_equivalents + accounts_receivable

                try: quick_ratio = quick_assets / total_current_liabilities
                except: quick_ratio = 'N/A (Cannot divide by zero)'

                try: inventory_turnover_ratio = cost_of_goods_sold / average_inventory
                except: inventory_turnover_ratio = 'N/A (Cannot divide by zero)'

                try: inventory_turnover_ratio_by_day = 365 / inventory_turnover_ratio
                except: inventory_turnover_ratio_by_day = 'N/A (Cannot divide by zero)'

                try: accounts_receivable_turnover_ratio = net_sales / average_accounts_receivable
                except: accounts_receivable_turnover_ratio = 'N/A (Cannot divide by zero)'

                try: debt_to_equity_ratio = long_term_debt / total_equity
                except: debt_to_equity_ratio = 'N/A (Cannot divide by zero)'

                try: return_on_equity = net_income / total_equity
                except: return_on_equity = 'N/A (Cannot divide by zero)'

                try: debt_equity_ratio = total_liabilities / total_equity
                except: debt_equity_ratio = 'N/A (Cannot divide by zero)'


                def view_report():
                    # Display the calculated results

                    st.markdown("#### Assets")
                    st.markdown(f'**Current Assets:**')

                    st.markdown(f'&nbsp;&nbsp;&nbsp;&nbsp;Accounts receivable and prepayments: {accounts_receivable_and_prepayments:,.2f}')
                    st.markdown(f'&nbsp;&nbsp;&nbsp;&nbsp;Cash and cash equivalents: {cash_and_cash_equivalents:,.2f}')
                    st.markdown(f'&nbsp;&nbsp;&nbsp;&nbsp;Inventory: {inventory:,.2f}')
                    st.markdown(f'&nbsp;&nbsp;&nbsp;&nbsp;Other short-term assets: {other_short_term_assets:,.2f}')
                    st.markdown(f'**Total Current Assets:** {total_current_assets:,.2f}')
                    st.markdown("""<hr style='border:1px solid blue'> """, unsafe_allow_html=True)
                    #
                    # st.markdown(f'**Non Current Assets:**')

                    st.markdown(f'&nbsp;&nbsp;&nbsp;&nbsp;Property, Plant & Equipment (PPE): {property_plant_equipment:,.2f}')
                    st.markdown(f'&nbsp;&nbsp;&nbsp;&nbsp;Long-term investments: {long_term_investments:,.2f}')
                    st.markdown(f'&nbsp;&nbsp;&nbsp;&nbsp;Intangible assets: {intangible_assets:,.2f}')
                    st.markdown(f'&nbsp;&nbsp;&nbsp;&nbsp;Deferred Charges and Other Noncurrent Assets: {deferred_charges:,.2f}')
                    st.markdown(f'**Total Non-current Assets:** {total_non_current_assets:,.2f}')

                    st.markdown(f'**Total Assets:** {total_assets:,.2f}')
                    st.markdown("""<hr style='border:1px solid blue'> """, unsafe_allow_html=True)

                    st.markdown("#### Liabilities and Equity")
                    st.markdown(f'**Liabilities:**')
                    st.markdown(f'**Current Liabilities:**')

                    st.markdown(f'&nbsp;&nbsp;&nbsp;&nbsp;Current portion of long-term debt: {current_portion_of_long_term_debt:,.2f}')
                    st.markdown(f'&nbsp;&nbsp;&nbsp;&nbsp;Short-term debt: {short_term_debt:,.2f}')
                    st.markdown(f'&nbsp;&nbsp;&nbsp;&nbsp;Account Payable: {accounts_payable:,.2f}')
                    st.markdown(f'**Total Current Liabilities:** {total_current_liabilities:,.2f}')
                    st.markdown("""<hr style='border:1px solid blue'> """, unsafe_allow_html=True)

                    st.markdown(f'**Noncurrent Liabilities:**')

                    st.markdown(f'&nbsp;&nbsp;&nbsp;&nbsp;Long-term Debt (loan): {long_term_debt_loan:,.2f}')
                    st.markdown(f'**Total Non-current Liabilities:** {total_non_current_liabilities:,.2f}')
                    st.markdown(f'**Total Liabilities:** {total_liabilities:,.2f}')
                    st.markdown("""<hr style='border:1px solid blue'> """, unsafe_allow_html=True)


                    st.markdown(f'**Equity:**')
                    st.markdown(f'&nbsp;&nbsp;&nbsp;&nbsp;Owner\'s Capital: {owner_capital:,.2f}')
                    st.markdown(f'&nbsp;&nbsp;&nbsp;&nbsp;Retained Earnings:{retained_earnings:,.2f}')
                    st.markdown(f'**Total Equity:** {total_equity:,.2f}')
                    st.markdown("<hr style='border:1px solid blue'> ", unsafe_allow_html=True)
                    st.markdown(f'**Total Liabilities & Equity:** {total_liabilities_and_equity:,.2f}')
                    st.markdown("<hr style='border:1px solid blue'> ", unsafe_allow_html=True)

                    # other_total = opening_accounts_receivable + opening_balance_ppe + opening_balance_inventory + net_sales + cost_of_goods_sold + net_income

                    # Display the additional variables
                    st.markdown("#### Other")
                    st.markdown(f'&nbsp;&nbsp;&nbsp;&nbsp;Opening accounts receivable: {opening_accounts_receivable:,.2f}')
                    st.markdown(f'&nbsp;&nbsp;&nbsp;&nbsp;Closing accounts receivable: {closing_accounts_receivable:,.2f}')
                    st.markdown(f'&nbsp;&nbsp;&nbsp;&nbsp;Opening balance PPE: {opening_balance_ppe:,.2f}')
                    st.markdown(f'&nbsp;&nbsp;&nbsp;&nbsp;Closing balance PPE: {closing_balance_ppe:,.2f}')
                    st.markdown(f'&nbsp;&nbsp;&nbsp;&nbsp;Opening balance inventory: {opening_balance_inventory:,.2f}')
                    st.markdown(f'&nbsp;&nbsp;&nbsp;&nbsp;Closing balance inventory: {closing_balance_inventory:,.2f}')
                    st.markdown(f'&nbsp;&nbsp;&nbsp;&nbsp;Net sales: {net_sales:,.2f}')
                    st.markdown(f'&nbsp;&nbsp;&nbsp;&nbsp;Cost of goods sold: {cost_of_goods_sold:,.2f}')
                    st.markdown(f'&nbsp;&nbsp;&nbsp;&nbsp;Net income: {net_income:,.2f}')
                    st.markdown("<hr style='border:1px solid blue'> ", unsafe_allow_html=True)

                    st.markdown(f'**Average PPE:** {average_ppe:,.2f}')
                    st.markdown(f'**Average Inventory:** {average_inventory:,.2f}')
                    st.markdown(f'**Average Accounts Receivable:** {average_accounts_receivable:,.2f}')



                    def display_ratio(ratio_name, ratio_value):
                        if isinstance(ratio_value, str):
                            st.markdown(f'<p style="color:red;">**Error calculating {ratio_name}:** {ratio_value}</p>',
                                        unsafe_allow_html=True)
                        else:
                            st.markdown(f'**{ratio_name}:** {ratio_value:,.2f}')


                    # Create a dictionary mapping ratio names to their calculated values
                    ratios = {
                        'Fixed Assets Turnover Ratio': fixed_assets_turnover_ratio,
                        'Quick Assets': quick_assets,
                        'Current Ratio': current_ratio,
                        'Inventory Turnover Ratio': inventory_turnover_ratio,
                        'Inventory Turnover Ratio by Day': inventory_turnover_ratio_by_day,
                        'Accounts Receivable Turnover Ratio': accounts_receivable_turnover_ratio,
                        'Debt to Equity Ratio': debt_to_equity_ratio,
                        'Return on Equity (ROE)': return_on_equity,
                        'Debt Equity Ratio': debt_equity_ratio,
                    }
                    # Loop through the dictionary and display each ratio
                    for ratio_name, ratio_value in ratios.items():
                        display_ratio(ratio_name, ratio_value)


                def view_quick_analysis_corp():
                    if not isinstance(fixed_assets_turnover_ratio, str) and \
                            not isinstance(current_ratio, str) and \
                            not isinstance(quick_ratio, str) and \
                            not isinstance(inventory_turnover_ratio, str) and \
                            not isinstance(inventory_turnover_ratio_by_day, str) and \
                            not isinstance(accounts_receivable_turnover_ratio, str) and \
                            not isinstance(debt_to_equity_ratio, str) and \
                            not isinstance(return_on_equity, str) and \
                            not isinstance(debt_equity_ratio, str):

                        st.markdown(f"- For each Riyal invested in fixed assets, the company generates {fixed_assets_turnover_ratio:.2f} in net sales.")
                        if current_ratio >= 1: st.markdown(f"- With a Current Ratio of {current_ratio:.2f}, the company is well-equipped to meet short-term liabilities with its short-term assets.")
                        else: st.markdown(f"- With a Current Ratio of {current_ratio:.2f}, the company may face difficulties in meeting short-term liabilities with its short-term assets.")
                        if quick_ratio >= 1: st.markdown(f"- A Quick Ratio of {quick_ratio:.2f} indicates the company's solid ability to cover current liabilities with its most liquid assets.")
                        else: st.markdown(f"- A Quick Ratio of {quick_ratio:.2f} suggests the company might have issues covering current liabilities with its most liquid assets.")
                        st.markdown(f"- The company rotates its inventory into finished goods approximately {inventory_turnover_ratio:.2f} times per year, translating to an average of {inventory_turnover_ratio_by_day:.2f} days to convert inventories into saleable products.")
                        st.markdown(f"- On average, the company collects its receivables {accounts_receivable_turnover_ratio:.2f} times per year.")
                        st.markdown(f"- The company has {debt_to_equity_ratio:.2f} units of debt for each unit of equity, indicating its leverage position.")
                        st.markdown(f"- The company generates a profit of {return_on_equity:.2f} per unit of equity, illustrating the return it makes on its own net resources.")
                        st.markdown(f"- For each unit of equity owned by the company's shareholders, the company owes {debt_equity_ratio:.2f} to its creditors, indicating its debt load relative to its equity.")


                def view_smart_analysis_corp():
                    if not isinstance(fixed_assets_turnover_ratio, str) and \
                            not isinstance(current_ratio, str) and \
                            not isinstance(quick_ratio, str) and \
                            not isinstance(inventory_turnover_ratio, str) and \
                            not isinstance(inventory_turnover_ratio_by_day, str) and \
                            not isinstance(accounts_receivable_turnover_ratio, str) and \
                            not isinstance(debt_to_equity_ratio, str) and \
                            not isinstance(return_on_equity, str) and \
                            not isinstance(debt_equity_ratio, str):
                        # Generate a single, comprehensive prompt
                        prompt = f"""
                        The company's financial performance is as follows:
    
                        - Fixed Assets Turnover Ratio: {fixed_assets_turnover_ratio:.2f}
                        - Current Ratio: {current_ratio:.2f}
                        - Quick Ratio: {quick_ratio:.2f}
                        - Inventory Turnover Ratio: {inventory_turnover_ratio:.2f}
                        - Inventory Turnover Ratio by Day: {inventory_turnover_ratio_by_day:.2f}
                        - Accounts Receivable Turnover Ratio: {accounts_receivable_turnover_ratio:.2f}
                        - Debt to Equity Ratio: {debt_to_equity_ratio:.2f}
                        - Return on Equity (ROE): {return_on_equity:.2f}
                        - Debt Equity Ratio: {debt_equity_ratio:.2f}
    
                        Based on this information, what insights and analyses can be drawn?
                        """
                        # Generate a single AI response
                        response = generate_prompt(prompt)

                        # Display the AI response
                        st.markdown(response)


                import time


                # Use the second and third columns to create the selectbox and button
                st.markdown("---")
                st.subheader("Report")
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    option_auto = st.selectbox('Please choose an analysis type',
                                          ('Statistics', 'Quick Analysis', 'Smart Analysis'), key='select_box_2')

                    if st.button('View', key='select_button_2'):
                        if option_auto == 'Statistics':
                            view_report()
                        elif option_auto == 'Quick Analysis':
                            with st.spinner('Wait ...'):
                                time.sleep(1.5)
                                view_quick_analysis_corp()
                        elif option_auto == 'Smart Analysis':
                            with st.spinner('Wait ...'):
                                time.sleep(1)
                                view_smart_analysis_corp()
                                time.sleep(3)

            else:
                st.error("Invalid file format. Please upload a CSV file with 'Description' and 'Amount' columns.")







      
                  


        

    
        
    
    
    
    
    
    
    
    
    
    