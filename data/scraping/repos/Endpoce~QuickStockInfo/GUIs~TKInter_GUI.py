import tkinter as tk
from tkinter import ttk
from datetime import datetime
import yfinance as yf
import pandas as pd
import openai
from Gen_Files.Company_Info_Web_Scraper import get_company_info, get_wiki_info, summarize_article
from Gen_Files.GetArticles import get_MW_Articles
from Gen_Files.Stock_Analyzer import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

yf.pdr_override()

# set openai api key
openai.api_key = os.environ.get('API_KEY')


def get_stock_data(symbol, start_date, end_date):
    ticker = yf.download(symbol, start_date, end_date)

    # save stock data to csv
    file = ticker.to_csv("Price_data\\"+symbol + '_Price_Data.csv')

    return file


def main():

    # Set default start and end dates
    default_start_date = "2020-01-01"
    default_end_date = datetime.today().strftime('%Y-%m-%d')

    def fetch_data():
        # Get user input
        ticker_symbol = ticker_entry.get().upper()
        start_date = start_date_entry.get()
        end_date = end_date_entry.get()

        # download and save stock data
        stock_data = get_stock_data(ticker_symbol, start_date, end_date)

        # display stock data
        filename = "Price_Data\\" + ticker_symbol + '_Price_Data.csv'

        data = pd.read_csv(filename, index_col=0, parse_dates=True)

        fig = plot_stock_with_moving_averages_from_csv(filename)

        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.get_tk_widget().pack()

        # Get company info and stock data
        info = get_company_info(ticker_symbol)

        info_label1 = tk.Grid(root, text=info['name'])
        info_label1.grid(row=2, column=0)
        info_label2 = tk.Label(root, text=info['sector'])
        info_label2.pack()
        info_label3 = tk.Label(root, text=info['industry'])
        info_label3.pack()
        info_label4 = tk.Label(root, text=info['summary'])
        info_label4.config(wraplength=500)
        info_label4.pack()

        # get wiki info
        wiki_url = get_wiki_info(ticker_symbol)
        wiki_url_label['text'] = wiki_url

        # read stock price data from csv
        filename = "Price_Data\\" + ticker_symbol + '_Price_Data.csv'
        df = pd.read_csv(filename)

        # analyze stock data
        analyze_result = analyze_stock(filename)
        analyze_label['text'] = analyze_result

        analyze_label.insert(tk.END, analyze_result)

        # get articles
        articles = get_MW_Articles(ticker_symbol)

        articles_label.insert(tk.END, '\n'.join(
            [article['title'], article['url'], summarize_article(article)]))

        # display articles
        for article in articles:
            articles_label['text'] = '\n'.join(
                [article['title'], article['url'], summarize_article(article)])

    root = tk.Tk()

    # Create frames
    input_frame = tk.Frame(root)
    input_frame.pack(side="top", fill="x")

    output_frame = tk.Frame(root)
    output_frame.pack(side="bottom", fill="both", expand=True)

    # User Input
    ticker_label = tk.Label(input_frame, text="Enter Ticker Symbol:")
    ticker_entry = tk.Entry(input_frame)
    start_date_label = tk.Label(input_frame, text="Start date:")
    start_date_entry = tk.Entry(input_frame)
    end_date_label = tk.Label(input_frame, text="End date:")
    end_date_entry = tk.Entry(input_frame)

    # Set default values for start and end date entries
    start_date_entry.insert(tk.END, default_start_date)
    end_date_entry.insert(tk.END, default_end_date)

    # Position widgets using grid
    ticker_label.grid(row=0, column=0)
    ticker_entry.grid(row=0, column=1)
    start_date_label.grid(row=1, column=0)
    start_date_entry.grid(row=1, column=1)
    end_date_label.grid(row=2, column=0)
    end_date_entry.grid(row=2, column=1)

    fetch_button = ttk.Button(
        input_frame, text="Get Stock Data", command=fetch_data)
    fetch_button.grid(row=3, column=0, columnspan=2)

    # Create labels in the output frame

    wiki_url_label = tk.Label(output_frame, text="")
    wiki_url_label.pack()
    analyze_label = tk.Label(output_frame, text="")
    analyze_label.pack()
    articles_label = tk.Label(output_frame, text="")
    articles_label.pack()

    root.mainloop()


if __name__ == "__main__":
    main()
