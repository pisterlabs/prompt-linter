import helper
import logging
from matplotlib import pyplot as plt
from fpdf import FPDF
import graphing
import os
from openai import OpenAI

client = OpenAI()

# === Documentation of pdf.py ===

def run(message, bot):
    """
    run(message, bot): This is the main function used to implement the pdf save feature.
    """
    try:
        helper.read_json()
        chat_id = message.chat.id
        user_history = helper.getUserHistory(chat_id)
        msg = "Crunching data"
        bot.send_message(chat_id, msg)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        top = 0.8
        if len(user_history) == 0:
            plt.text(
                0.1,
                top,
                "No record found!",
                horizontalalignment="left",
                verticalalignment="center",
                transform=ax.transAxes,
                fontsize=20,
            )
        for rec in user_history:
            date, category, amount = rec.split(",")
            print(date, category, amount)
            rec_str = f"{amount}$ {category} expense on {date}"
            plt.text(
                0,
                top,
                rec_str,
                horizontalalignment="left",
                verticalalignment="center",
                transform=ax.transAxes,
                fontsize=14,
                bbox=dict(facecolor="red", alpha=0.3),
            )
            top -= 0.15
        plt.axis("off")
        plt.savefig("expense_history.png")
        plt.close()

        if helper.isCategoryBudgetAvailable(chat_id):
            category_budget = {}
            for cat in helper.spend_categories:
                if helper.isCategoryBudgetByCategoryAvailable(chat_id, cat):
                    category_budget[cat] = helper.getCategoryBudgetByCategory(chat_id, cat)
            graphing.overall_split(category_budget)

        category_spend = {}
        for cat in helper.spend_categories:
            spend = helper.calculate_total_spendings_for_cateogory_chat_id(chat_id,cat)
            if spend != 0:
                category_spend[cat] = spend
        if category_spend != {}:
            graphing.spend_wise_split(category_spend)

        if helper.isCategoryBudgetAvailable(chat_id):
            category_spend_percent = {}
            for cat in helper.spend_categories:
                if helper.isCategoryBudgetByCategoryAvailable(chat_id, cat):
                    percent = helper.calculateRemainingCateogryBudgetPercent(chat_id, cat)
                    category_spend_percent[cat] = percent
            graphing.remaining(category_spend_percent)

        if helper.getUserHistory(chat_id):
            cat_spend_dict = helper.getUserHistoryDateExpense(chat_id)
            graphing.time_series(cat_spend_dict)
        
        list_of_images = ["overall_split.png","spend_wise.png","remaining.png","time_series.png"]

        messages = [
            {"role": "user", "content": "You are a specialized finance advisor bot. You must the budget items I provide and do an analysis on them"},
            {"role": "user", "content": "Keep the analysis short"},
            {"role": "user", "content": "Highlight any potential possibilites where I can save money or where I am wasting money"},
            {"role": "user", "content": "My budget is as below"},
        ]
        chat_id = message.chat.id
        data = helper.getUserData(chat_id)

        print(data)

        expenses = data['expense']
        expenses_str = ''
        if expenses:
            for exp in expenses:
                temp = exp.split(',')
                formatted = f"Expense: Date: {temp[0]} Category: {temp[1]} Amount: {temp[2]}"
                expenses_str = f"{expenses_str}\n{formatted}"

        incomes = data['income']
        incomes_str = ''
        if incomes:
            for exp in incomes:
                temp = exp.split(',')
                formatted = f"Income: Date: {temp[0]} Name: {temp[1]} Amount: {temp[2]}"
                incomes_str = f"{incomes_str}\n{formatted}"

        # recurrent = data['recurrent']

        balance = data['budget']['budget']
        savings = data['budget']['savings']

        balance_str = f"Balance in bank: {balance}"
        saving_str = f"Balance in savings account: {savings}"

        if(expenses_str):
            messages.append({"role": "user", "content": f"{expenses_str}"})
        if(incomes_str):
            messages.append({"role": "user", "content": f"{incomes_str}"})
        if(balance_str):
            messages.append({"role": "user", "content": f"{balance_str}"})
        if(saving_str):
            messages.append({"role": "user", "content": f"{saving_str}"})

        bot.send_message(chat_id, "Starting AI analysis")
        print('called analyse')
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=messages,
        )
        ai_report = response.choices[0].message.content

        pdf = FPDF()
        pdf.add_page()

        pdf.set_font("Arial", 'B', 18)
        pdf.cell(200, 10, txt="Analysis", ln=True, align='C')

        # line_height = 6  
        # pdf.set_line_height(line_height)

        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 4, txt=ai_report)


        x_coord = 20
        y_coord = pdf.get_y() + 10

        for image in list_of_images:
            pdf.image(image,x=x_coord,y=y_coord,w=70,h=50)
            x_coord += 80
            if x_coord > 100:
                x_coord = 20
                y_coord += 60
        pdf.output("expense_report.pdf", "F")


        bot.send_document(chat_id, open("expense_report.pdf", "rb"))
        for image in list_of_images:
            os.remove(image)
    except Exception as e:
        logging.exception(str(e))
        bot.reply_to(message, "Oops!" + str(e))