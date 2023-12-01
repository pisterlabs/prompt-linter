"""
        INSERT INTO Users (email, password)
        VALUES (?, ?)
    """"""
        SELECT * FROM Users WHERE email = ? AND password = ?
    """"""
        SELECT user_id FROM Users WHERE email = ?
    """"""
            INSERT INTO Transcripts (user_id, file_name, transcription, transcription_summary) 
            VALUES (?, ?, ?, ?)
        """"""
        INSERT INTO Users (email, password)
        VALUES (?, ?)
    """"""
        SELECT * FROM Users WHERE email = ? AND password = ?
    """"""
        SELECT user_id FROM Users WHERE email = ?
    """"""
            INSERT INTO Transcripts (user_id, file_name, transcription, transcription_summary) 
            VALUES (?, ?, ?, ?)
        """'''
        You are a helpful AI assistant, intended to fix any spelling or grammar mistakes in user audio transcript.
        \nIf words appear incorrect or there are run-on word, fix the transcript the best you can.   
    '''f'''
                                Fact-check this transcript for factual or logical inacurracies or inconsistencies
                                \nWrite a report on the factuality / logic of the transcirpt
                                \nTRANSCRIPT: {st.session_state.transcript}
                                \nTRANSCRIPT SUMMARY: {st.session_state.transcript_summary}
                                \nAI FACT CHECK RESPONSE HERE:
                        ''''''
        Fact-check this transcript for factual or logical inacurracies or inconsistencies
        \nWrite a report on the factuality / logic of the transcirpt
        \nTRANSCRIPT: {}
        \nTRANSCRIPT SUMMARY: {}
        \nAI FACT CHECK RESPONSE HERE:
'''f'''
                                Fact-check this transcript for factual or logical inacurracies or inconsistencies
                                \nWrite a report on the factuality / logic of the transcirpt
                                \nTRANSCRIPT: {st.session_state.transcript}
                                \nTRANSCRIPT SUMMARY: {st.session_state.transcript_summary}
                                \nAI FACT CHECK RESPONSE HERE:
                        '''"""
            INSERT INTO Transcripts (file_name, transcription, transcription_summary) 
            VALUES (?, ?, ?)
        """