import threading
import openai
import logging

logger = logging.getLogger(__name__)
import time


class MyThread(threading.Thread):
    def __init__(self, thread_id, llm, account_manager, inp_manager, print_error, pbar, turbo_system_message,
                 print_finish=True):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.openai_account_manager_multi_thread = account_manager
        self.openai_inp_manager = inp_manager
        self.account = self.openai_account_manager_multi_thread.get_next_account(self.thread_id)
        self.print_error = print_error
        self.pbar = pbar
        self.print_finish = print_finish
        self.turbo_system_message = turbo_system_message
        self.llm = llm

    def run(self):

        def repeat_until_success_call_openai_api(func):
            def wrapper(*args, **kw):
                while 1:
                    result = None
                    try:
                        result = func(*args, **kw)
                    except openai.error.APIConnectionError as e:
                        if self.print_error:
                            logger.info('openai connection error, so retry after sleep 5 seconds')
                            logger.info(e)
                        time.sleep(5)
                    except openai.error.RateLimitError as e:
                        logger.info(type(e))
                        if 'quota' in e._message:
                            if self.print_error:
                                logger.info('now openai account {} runs out. so use next.'.format(self.account[-1]))
                                logger.info(type(e))
                                logger.info(e)
                            self.account = self.openai_account_manager_multi_thread.get_next_account(self.thread_id,
                                                                                                     self.account)
                        else:
                            logger.info("Meeting RateLimitError, sleep for 45 seconds.")
                            time.sleep(45)
                    except openai.error.AuthenticationError as e:
                        if 'This key is associated with a deactivated account' in e._message:
                            logger.info('the account {} is deactivated. so use next'.format(self.account[-1]))
                            if self.print_error:
                                logger.info(e)
                            self.account = self.openai_account_manager_multi_thread.get_next_account(self.thread_id,
                                                                                                     self.account)
                        else:
                            logger.info('meet unexpected AuthenticationError, so retry after sleep 5 seconds')
                            if self.print_error:
                                logger.info(e)
                            self.account = self.openai_account_manager_multi_thread.get_next_account(self.thread_id,
                                                                                                     self.account)
                    except Exception as e:
                        logger.info('meet unexpected error, so retry after sleep 5 seconds')
                        logger.info(e)
                        logger.info(type(e))
                        time.sleep(5)

                    if result != None:
                        return result
                    else:
                        pass

            return wrapper

        # pbar = tqdm.tqdm(total=len(self.idx_x_list_to_decode))
        responses_with_idx = []
        self.responses_with_idx = responses_with_idx
        while True:
            tmp = self.openai_inp_manager.get_next_gpt_idx_inp()
            if tmp == None:
                if self.print_finish:
                    logger.info('thread {} finish'.format(self.thread_id))
                return
            else:
                idx_inp = tmp['inp']
                idx, inp = idx_inp
                hyper_parameter = tmp['hyper_parameter']

                @repeat_until_success_call_openai_api
                def tmp_api_call():

                    result = self.llm.generate(inp, hyper_parameter['max_tokens'], api_key=self.account[-1],
                                               turbo_system_message=self.turbo_system_message)
                    return result

                response = tmp_api_call()
                if self.pbar is not None:
                    self.pbar.update(1)
                responses_with_idx.append([idx, response])
