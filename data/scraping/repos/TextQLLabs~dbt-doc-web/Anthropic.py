from typing import Optional, Dict, List
import anthropic
from API_keys.apikey import anthropicKey

class OAIRequest:
    def __init__(self, model: str, prompt: str, temperature: float, max_tokens: int):
        self.model = model
        self.prompt = prompt
        self.temperature = temperature
        self.max_tokens = max_tokens

class OAIRequestWithUserInfo:
    def __init__(self, prompt: str, email: str):
        self.prompt = prompt
        self.email = email

class OAIChoice:
    def __init__(self, text: str):
        self.text = text

class OAIResponse:
    def __init__(self, choices: List[OAIChoice]):
        self.choices = choices

class KeyOrUserInfo:
    def __init__(self, key: Optional[str] = None, user_info: Optional[str] = None):
        self.key = key
        self.user_info = user_info

class ArguParseException(Exception):
    def __init__(self, message):
        super().__init__(message)

class Depends:
    def __init__(self, nodes: Optional[List[str]] = None, macros: Optional[List[str]] = None):
        self.nodes = nodes
        self.macros = macros

class Env:
    def __init__(
        self,
        api_key: KeyOrUserInfo,
        base_path: str,
        project_name: str,
        models: Optional[set[str]] = None,
        dry_run: bool = False,
    ):
        self.api_key = api_key
        self.base_path = base_path
        self.project_name = project_name
        self.models = models
        self.dry_run = dry_run

class ColumnMetadata:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

class NodeMetadata:
    def __init__(
        self,
        original_file_path: str,
        patch_path: Optional[str] = None,
        compiled_code: Optional[str] = None,
        raw_code: Optional[str] = None,
        description: Optional[str] = None,
        database: Optional[str] = None,
        schema: Optional[str] = None,
        resource_type: Optional[str] = None,
        package_name: Optional[str] = None,
        path: Optional[str] = None,
        alias: Optional[str] = None,
        checksum: Optional[str] = None,
        config: Optional[str] = None,
        tags: Optional[str] = None,
        meta: Optional[str] = None,
        group: Optional[str] = None,
        docs: Optional[str] = None,
        build_path: Optional[str] = None,
        deferred: Optional[str] = None,
        unrendered_config: Optional[str] = None,
        created_at: Optional[str] = None,
        name: str = None,
        unique_id: str = None,
        fqn: List[str] = None,
        columns: Dict[str, ColumnMetadata] = None,
        depends_on: Optional[Depends] = None,
    ):
        self.original_file_path = original_file_path
        self.patch_path = patch_path
        self.compiled_code = compiled_code
        self.raw_code = raw_code
        self.description = description
        self.database = database
        self.schema = schema
        self.resource_type = resource_type
        self.package_name = package_name
        self.path = path
        self.alias = alias
        self.checksum = checksum
        self.config = config
        self.tags = tags
        self.meta = meta
        self.group = group
        self.docs = docs
        self.build_path = build_path
        self.deferred = deferred
        self.unrendered_config = unrendered_config
        self.created_at = created_at
        self.name = name
        self.unique_id = unique_id
        self.fqn = fqn
        self.columns = columns
        self.depends_on = depends_on

class Anthropic:

    metricsGuidelines = """General Information:
        A metric is an aggregation over a table that supports zero or more dimensions.
        Metric names must:
            contain only letters, numbers, and underscores (no spaces or special characters)
            begin with a letter
            contain no more than 250 characters


        Available properties (Define aspects of the metric)
        Field: name
        Description: A unique identifier for the metric
        Example: new_customers
        ---
        Field: model
        Description: The dbt model that powers this metric
        Example: dim_customers
        ---
        Field: label
        Description: A short for name / label for the metric
        Example: New Customers
        ---
        Field: description
        Description: Long form, human-readable description for the metric
        Example: The number of customers who....
        ---
        Field: calculation_method
        Description: The method of calculation (aggregation or derived) that is applied to the expression
        Example: count_distinct
        ---
        Field: expression
        Description: The expression to aggregate/calculate over
        Example: user_id, cast(user_id as int)
        Guidelines: You can't use * as expression in the metric. It can be used only as a SQL expression (for instance count(*))
                    For expressions that needs math operations (sum,average,median) make sure you are using a numeric column (acoording the provided data type of the column).
        ---
        Field: timestamp
        Description: 
        Example: signup_date
        ---
        Field: time_grains
        Description: One or more "grains" at which the metric can be evaluated. For more information, see the "Custom Calendar" section.
        Example: [day, week, month, quarter, year]
        ---
        Field: dimensions
        Description: A list of dimensions to group or filter the metric by
        Example: [plan, country]
        ---
        Field: window
        Description: A dictionary for aggregating over a window of time. Used for rolling metrics such as 14 day rolling average. Acceptable periods are: [day,week,month, year, all_time]
        Example: {{count: 14, period: day}}
        ---
        Field: filters
        Description: A list of filters to apply before calculating the metric
        Guideline: Filters should be defined as a list of dictionaries that define predicates for the metric. Filters are combined using AND clauses. For more control, users can (and should) include the complex logic in the model powering the metric.
                All three properties (field, operator, value) are required for each defined filter.
        ---
        


        Available calculation methods (The method of calculation (aggregation or derived) that is applied to the expression)
        Method: count
        Description: This metric type will apply the count aggregation to the specified field
        ---
        Method: count_distinct
        Description: This metric type will apply the count aggregation to the specified field, with an additional distinct statement inside the aggregation
        ---
        Method: sum
        Description: This metric type will apply the sum aggregation to the specified field
        ---
        Method: average
        Description: This metric type will apply the average aggregation to the specified field
        ---
        Method: min
        Description: This metric type will apply the min aggregation to the specified field
        ---
        Method: max
        Description: This metric type will apply the max aggregation to the specified field
        ---
        Method: median
        Description: This metric type will apply the median aggregation to the specified field, or an alternative percentile_cont aggregation if median is not available
        ---
        Method: derived
        Description: This metric type is defined as any non-aggregating calculation of 1 or more metrics
        ---
        """

    async def run_request(prompt: str) -> str:
        resGPT=""        
        
        try:
            client = anthropic.AsyncAnthropic(api_key=anthropicKey)
            
            completion = await client.completions.create(
                model="claude-2",
                max_tokens_to_sample=1000,
                prompt=f"\n\nHuman: {prompt}\n\nAssistant:",
            )
            print(completion.completion)

            resGPT = str(completion.completion)

        except Exception as e:
            print("Error in run_request: " + str(e))            

        return resGPT  

        # def mk_prompt(reverse_deps: Dict[str, List[str]], node: NodeMetadata) -> str:
    #     deps = ",".join(node.depends_on.nodes) if node.depends_on and node.depends_on.nodes else "(No dependencies)"
    #     r_deps = (
    #         ",".join(reverse_deps[node.unique_id])
    #         if node.unique_id in reverse_deps
    #         else "Not used by any other models"
    #     )
    #     staging = "\nThis is a staging model. Be sure to mention that in the summary.\n" if "staging" in node.fqn else ""
    #     raw_code = node.raw_code if node.raw_code else ""

    #     prompt = f"""Write markdown documentation to explain the following DBT model. Be clear and informative, but also accurate. The only information available is the metadata below.
    #     Explain the raw SQL, then explain the dependencies. Do not list the SQL code or column names themselves; an explanation is sufficient.

    #     Model name: {node.name}
    #     Raw SQL code: {raw_code}
    #     Depends on: {deps}
    #     Depended on by: {r_deps}
    #     {staging}
    #     First, generate a human-readable name for the table as the title (i.e. fct_orders -> # Orders Fact Table).
    #     Then, describe the dependencies (both model dependencies and the warehouse tables used by the SQL.) Do this under ## Dependencies.
    #     Then, describe what other models reference this model in ## How it's used
    #     Then summarize the model logic in ## Summary.
    #     """
    #     return prompt
    
    async def mk_prompt(raw_code: str) -> str:
        prompt = f"""Write markdown documentation to explain the following DBT model. Be clear and informative, but also accurate. The only information available is the metadata below.
        Explain the raw SQL, then explain the dependencies. Do not list the SQL code or column names themselves; an explanation is sufficient.
        
        Raw SQL code: {raw_code}
        First, generate a human-readable name for the table as the title (i.e. fct_orders -> # Orders Fact Table).
        Then, describe the dependencies (both model dependencies and the warehouse tables used by the SQL.) Do this under ## Dependencies.
        Then, describe what other models reference this model in ## How it's used
        Then summarize the model logic in ## Summary.
        """
        return prompt

    def mk_column_prompt(node: NodeMetadata, col: ColumnMetadata, documented_nodes: Dict[str, NodeMetadata]) -> str:
        # Check if the current column depends on any documented nodes
        deps = node.depends_on.nodes if node.depends_on and node.depends_on.nodes else []
        inherited_docs = []
        for dep in deps:
            if dep in documented_nodes:
                # Add the name and description of each column in the dependent node to the list of inherited docs
                for dep_col in documented_nodes[dep].columns.values():
                    if dep_col.name in dep_col.depends_on.columns:
                        inherited_docs.append(dep_col.description)

        # Combine the inherited docs into a single string
        inherited_docs_str = "\n\n".join(inherited_docs)
        inheritance = f"""This column is inherited from another model. Use this column's documentation from the original model 
        as context for writing the requested one and be sure to mention it alongside the name of the original model. 
        Inherited documentation: {inherited_docs_str} """ if inherited_docs_str else ""

        prompt = f"""Write markdown documentation to explain the following DBT column in the context of the parent model and SQL code. Be clear and informative, but also accurate. The only information available is the metadata below.
        Do not list the SQL code or column names themselves; an explanation is sufficient.

        Column Name: {col.name}
        Parent Model name: {node.name}
        Raw SQL code: {node.raw_code}
        {inheritance}

        First, explain the meaning of the column in plain, non-technical English. Then, explain how the column is extracted in code.
        If the column is calculated from other columns, explain how the calculation works.
        If the column is derived from other columns, explain how those columns are extracted.
        If the column is a inherited from another model, mention the original model and use the provided Inherited documentation (If there is any).
        Remember not to list the SQL code, brackets, or the word config (Jinja code); an explanation is sufficient.
        """
        return prompt    

    # @staticmethod
    # def promptDesignMetrics(node: NodeMetadata, column_list: List[str]):
    #     prompt = f"""You are a data analyst that receives the structure of a DBT model/table and outputs statements for the generation of DBT metrics and dimension for the model/table.
    #     You are given the metadata of a DBT model and a list of columns and data types in that model.
    #     You return three statements separated by this character: ';'
    #     Remember, just 3 statements! and the final one should not have the character: ';' at the end.
    #     Each statement is a prompt for the generation of a DBT metric and/or dimension that will be done by another LLM. You also should return the name of the metric that you intend to create with the prompt.
    #     You should separate the name of the metric from the prompt with this character: ':'. Remember, the name of the metric should always be at the left side of the ':', and the prompt at the rigt one.
    #     The generated metrics should be diverse and helpful for the analysis of the model/table.
    #     These are some guidelines for the generation of metrics:
    #     {OpenAI.metricsGuidelines}

    #     These are some examples of generation of statements for the generation of metrics:

    #     Generic Example 1:

    #     Input:
    #     model: dim_customers
    #     model_description: A table containing all customers
    #     sql_query: SELECT * FROM dim_customers
    #     columns: ['customer_id', 'customer_name', 'customer_email', 'customer_phone', 'customer_address', 'customer_city', 'customer_state', 'customer_country', 'lifetime_value', 'is_paying', 'company_name', 'signup_date', 'plan']
        
    #     Output:
    #     new_customers:new customers using the product by month and half of the month;
    #     total_revenue:total revenue (lifetime_value) by the following dimensions customer_name, customer_country, and company_name;
    #     customers_count:number of customers by the following dimensions customer_city, customer_country, and company_name

    #     Generic Example 2:

    #     Input:
    #     model: order_events
    #     model_description: model used by the finance team to better understand revenue but inconsistencies in how it's reported have led to requests that the data team centralize the definition in the dbt repo.
    #     sql_query: SELECT * FROM order_events
    #     columns: ['event_date', 'order_id', 'order_country', 'order_status', 'customer_id', 'customer_status', 'amount']
        
    #     Output:
    #     total_revenue:Total revenue by year and country;
    #     total_orders:Total orders by customer status and country;
    #     average_revenue:Average revenue by year and country

    #     Complete the following. Return the three statements separated by ; with no ther comment.
    #     model: {node.name}
    #     model_description: {node.description}
    #     sql_query: {node.raw_code}
    #     columns: {column_list}    
    #     Output:
    #     """
    #     return prompt

    @staticmethod
    async def promptDesignMetrics(raw_code: str) -> str:
        prompt = f"""You are a data analyst that receives the structure of a DBT model/table and outputs statements for the generation of DBT metrics and dimension for the model/table.
        You are given the metadata of a DBT model and a list of columns and data types in that model.
        You return only one statement.
        Remember, just 1 statement!
        Each statement is a prompt for the generation of a DBT metric and/or dimension that will be done by another LLM. You also should return the name of the metric that you intend to create with the prompt.
        You should separate the name of the metric from the prompt with this character: ':'. Remember, the name of the metric should always be at the left side of the ':', and the prompt at the rigt one.
        The generated metrics should be diverse and helpful for the analysis of the model/table.
        These are some guidelines for the generation of metrics:
        {Anthropic.metricsGuidelines}

        These are some examples of generation of statements for the generation of metrics:

        Generic Example 1:

        Input:        
        sql_query: SELECT * FROM dim_customers        
        
        Output option A:
        new_customers:new customers using the product by month and half of the month
        Output option B:
        total_revenue:total revenue (lifetime_value) by the following dimensions customer_name, customer_country, and company_name
        Output option C:
        customers_count:number of customers by the following dimensions customer_city, customer_country, and company_name

        Generic Example 2:

        Input:        
        sql_query: SELECT * FROM order_events        
        
        Output option A:
        total_revenue:Total revenue by year and country
        Output option B:
        total_orders:Total orders by customer status and country
        Output option C:
        average_revenue:Average revenue by year and country

        Complete the following. Return one statement with no ther comment or code. Just metric_name:prompt.
        sql_query: {raw_code}        
        Output:
        """
        return prompt

#     @staticmethod
#     def promptGenMetrics(node: NodeMetadata, column_list: List[str], query: str, metricName: str):
#         prompt = f"""You are an expert SQL analyst with a large knowledge of the DBT platform that takes natural language input and outputs DBT metrics in YAML format.
#         You are given the metadata of a DBT model, the list of columns and data types in that model, and the name of the metric.
        
#         follow the following guidelines to generate the metric:
#         {OpenAI.metricsGuidelines}

#         Some examples of metric generation are:

#         Generic Example 1:

#         model: dim_customers
#         model_description: A table containing all customers
#         sql_query: SELECT * FROM dim_customers
#         columns: ['customer_id', 'customer_name', 'customer_email', 'customer_phone', 'customer_address', 'customer_city', 'customer_state', 'customer_country', 'lifetime_value', 'is_paying', 'company_name', 'signup_date', 'plan']
#         metric_name: new_customers
#         Input: new customers using the product in half of the month.
#         YAML Output:version: 2

# metrics:
#   - name: new_customers
#     label: New Customers
#     model: ref('dim_customers')
#     description: "The 14 day rolling count of paying customers using the product"

#     calculation_method: count_distinct
#     expression: customer_id 

#     dimensions:
#     - plan
#     - customer_country

#     filters:
#     - field: is_paying
#       operator: 'is'
#       value: 'true'
#     - field: lifetime_value
#       operator: '>='
#     value: '100'
#     - field: company_name
#       operator: '!='
#       value: "'Acme, Inc'"
#     - field: signup_date
#       operator: '>='
#       value: "'2020-01-01'"


#         Generic Example 2:

#         model: dim_customers
#         model_description: A table containing all customers
#         sql_query: SELECT * FROM dim_customers
#         columns: ['customer_id', 'customer_name', 'customer_email', 'customer_phone', 'customer_address', 'customer_city', 'customer_state', 'customer_country', 'lifetime_value', 'is_paying', 'company_name', 'signup_date', 'plan']
#         metric_name: average_revenue_per_customer
#         Input: average revenue per customer, segment data per plan and country.
#         YAML Output:version: 2

# metrics:
#   - name: average_revenue_per_customer
#     label: Average Revenue Per Customer
#     model: ref('dim_customers')
#     description: "The average revenue received per customer"

#     calculation_method: average
#     expression: lifetime_value 

#     dimensions:
#     - plan
#     - customer_country
            

#         Complete the following. Use YAML format and include no other commentary.
#         model: {node.name}
#         columns: {column_list}
#         model_description: {node.description}
#         sql_query: {node.raw_code}
#         metric_name: {metricName}
#         Input: {query}
#         YAML Output:"""
#         return prompt

    @staticmethod
    async def promptGenMetrics(query: str, statement: str, metricName: str):
        prompt = f"""You are an expert SQL analyst with a large knowledge of the DBT platform that takes natural language input and outputs DBT metrics in YAML format.
        You are given the metadata of a DBT model, the list of columns and data types in that model, and the name of the metric.
        
        follow the following guidelines to generate the metric:
        {Anthropic.metricsGuidelines}

        Some examples of metric generation are:

        Generic Example 1:
        
        sql_query: SELECT * FROM dim_customers
        metric_name: new_customers
        Input: new customers using the product in half of the month.
        YAML Output:version: 2

metrics:
  - name: new_customers
    label: New Customers
    model: ref('dim_customers')
    description: "The 14 day rolling count of paying customers using the product"

    calculation_method: count_distinct
    expression: customer_id 

    dimensions:
    - plan
    - customer_country

    filters:
    - field: is_paying
      operator: 'is'
      value: 'true'
    - field: lifetime_value
      operator: '>='
    value: '100'
    - field: company_name
      operator: '!='
      value: "'Acme, Inc'"
    - field: signup_date
      operator: '>='
      value: "'2020-01-01'"


        Generic Example 2:
        
        sql_query: SELECT * FROM dim_customers        
        metric_name: average_revenue_per_customer
        Input: average revenue per customer, segment data per plan and country.
        YAML Output:version: 2

metrics:
  - name: average_revenue_per_customer
    label: Average Revenue Per Customer
    model: ref('dim_customers')
    description: "The average revenue received per customer"

    calculation_method: average
    expression: lifetime_value 

    dimensions:
    - plan
    - customer_country
            

        Complete the following:       
        sql_query: {query}
        metric_name: {metricName}
        Input: {statement}
        YAML Output:"""
        return prompt

    async def promptGenMetricSQL(metric: str):
        prompt = f"""You are an expert SQL analyst with a large knowledge of the DBT platform that takes DBT metrics in YAML format as input and outputs queries in SQL format.
        You are given the definition and the name of the metric.

        Generic Example 1:
        
        metric_name: new_customers    
        metric:version: 2

    metrics:
    - name: new_customers
        label: New Customers
        model: ref('dim_customers')
        description: "The 14 day rolling count of paying customers using the product"

        calculation_method: count_distinct
        expression: customer_id 

        dimensions:
        - plan
        - customer_country

        filters:
        - field: is_paying
            operator: 'is'
            value: 'true'
        - field: lifetime_value
            operator: '>='
            value: '100'
        - field: company_name
            operator: '!='
            value: "'Acme, Inc'"
        - field: signup_date
            operator: '>='
            value: "'2020-01-01'"

        SQL Output:select * 
from {{{{ metrics.calculate(
    metric('new_customers'),    
    dimensions=['plan', 'customer_country']    
) }}}}


        Generic Example 2:
        
        metric_name: average_revenue_per_customer    
        metric:version: 2

    metrics:
    - name: average_revenue_per_customer
        label: Average Revenue Per Customer
        model: ref('dim_customers')
        description: "The average revenue received per customer"

        calculation_method: average
        expression: lifetime_value 

        dimensions:
        - plan
        - customer_country
            
        SQL Output:select * 
from {{{{ metrics.calculate(
    metric('average_revenue_per_customer'),    
    dimensions=['plan', 'customer_country']    
) }}}}

        Complete the following. Use SQL format and include no other commentary.
        metric: {metric}
        SQL Output:"""
        return prompt  

    def promptFixMetric(error: str, metric: str):
        prompt = f"""You are an expert SQL analyst with a large knowledge of the DBT platform that takes natural language input and outputs DBT metrics in YAML format.
        In a previous request you were given the metadata of a DBT model, the list of columns and data types in that model, and the name of the metric and you returned the content of the metric.
        The problem is that the metric is not correct and you need to fix it. You are given the error message and the metric.
        You need to fix the metric so that it follows the following guidelines:        
        {Anthropic.metricsGuidelines}
        You should return the content of the fixed metric in YAML format with no other comments or markers.
        Metric content: {metric}
        Metric error: {error}
        Fixed Metric content: """
        return prompt