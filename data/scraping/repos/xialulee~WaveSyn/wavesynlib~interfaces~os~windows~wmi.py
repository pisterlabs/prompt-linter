# -*- coding: utf-8 -*-
"""
Created on Sat Mar 04 21:02:17 2017

@author: Feng-cong Li
"""
import json
import abc
from collections import OrderedDict
from comtypes import client



class SWbemSink(metaclass=abc.ABCMeta):
    def __init__(self, *args, **kwrags):
        pass
    
    
    def ISWbemSinkEvents_OnObjectReady(self, this, objWbemObject, objWbemAsyncContext):
        self.on_object_ready(objWbemObject, objWbemAsyncContext)
    
    
    @abc.abstractmethod    
    def on_object_ready(self, wbem_object, context):
        pass
    
    
    def ISWbemSinkEvents_OnCompleted(self, this,
        iHResult, objWbemErrorObject, objWbemAsyncContext):
        self.on_completed(iHResult, objWbemErrorObject, objWbemAsyncContext)
        
        
    @abc.abstractmethod
    def on_completed(self, hresult, error_object, context):
        pass
    
    
    def ISWbemSinkEvents_OnProgress(self, this,
        iUpperBound, iCurrent, strMessage, objWbemAsyncContext):
        self.on_progress(iUpperBound, iCurrent, strMessage, objWbemAsyncContext)
        
        
    @abc.abstractmethod
    def on_progress(self, upper_bound, current, message, context):
        pass
    
    
    def ISWbemSinkEvents_OnObjectPut(self, this, objWbemObjectPath, objWbemAsyncContext):
        self.on_object_put(objWbemObjectPath, objWbemAsyncContext)
        
        
    @abc.abstractmethod
    def on_object_put(self, object_path, context):
        pass



class WQL:
    def __init__(self, services):
        self.__services = services
        self.__connection = None
        self.__com_sink = None
        
        
    def query(self, wql_str, output_format='original'):
        items = self.__services.ExecQuery(wql_str)

        def to_native(items):
            result = []
            for item in items:
                d = OrderedDict()
                for prop in item.Properties_:
                    d[prop.Name] = prop.Value
                result.append(d)
            return result
            
        def to_json(items):
            result = to_native(items)
            return json.dumps(result)
        
        def identity(items):
            return items
        
        return {
            'original': identity,
            'comtypes': identity,
            'native': to_native,
            'python': to_native,
            'json': to_json
        }[output_format](items)
        

    def gpt_query(self, prompt_str, output_format="python"):
        import openai
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"""\
Convert the following query to WMI query language:
{prompt_str} 
Only WQL code in your response.
"""}
            ]
        )       
        wql_str = response.choices[0].message.content
        return self.query(wql_str=wql_str, output_format=output_format)
        
        
        
    def set_sink(self, sink, wql_str):
        self.__com_sink = com_sink = client.CreateObject('WbemScripting.SWbemSink')
        py_sink = sink
        self.__connection = client.GetEvents(com_sink, py_sink)
        self.__services.ExecNotificationQueryAsync(com_sink, wql_str)
        