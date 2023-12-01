"""
        INSERT INTO Users (email, password)
        VALUES (?, ?)
    """"""
        SELECT * FROM Users WHERE email = ? AND password = ?
    """f'''
                You are an AI ChatBot intended to help with user stock data.
                \nYou have access to a pandas dataframe with the following specifications 
                \nDATA MODE: {metric_dropdown}
                \nSTOCKS: {asset_dropdown} 
                \nTIME PERIOD: {start} to {end}
                \nCHAT HISTORY: {st.session_state.chat_history}
                \nUSER MESSAGE: {query}
                \nAI RESPONSE HERE:
            '''f'''
            You are an AI ChatBot intended to help with user stock data.
            \nYou have access to a pandas dataframe with the following specifications 
            \nDATA MODE: {metric_dropdown}
            \nSTOCKS: {asset_dropdown} 
            \nTIME PERIOD: {start} to {end}
            \nCHAT HISTORY: {st.session_state.chat_history}
            \nUSER MESSAGE: {query}
            \nAI RESPONSE HERE:
        '''