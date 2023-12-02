import openai

class SQL_Traslator:
    def __init__(self, api_key) -> None:
        openai.api_key = api_key
        self._creation_db_script = """
            ```sql
                CREATE DATABASE nombre_de_tu_base_de_datos;

                \c nombre_de_tu_base_de_datos;

                CREATE TABLE product_category (
                    id INTEGER PRIMARY KEY,
                    name VARCHAR(100),
                    parent_id INTEGER, 
                    FOREIGN KEY (parent_id) REFERENCES product_category(id)
                );

                CREATE TABLE product_template (
                    id INTEGER PRIMARY KEY,
                    name VARCHAR(100),
                    type VARCHAR(100),
                    categ_id INTEGER,
                    list_price NUMERIC,
                    FOREIGN KEY (categ_id) REFERENCES product_category(id)
                );

                CREATE TABLE product_attribute (
                    id INTEGER PRIMARY KEY,
                    name VARCHAR(100)
                );

                CREATE TABLE product_attribute_value (
                    id INTEGER PRIMARY KEY,
                    name VARCHAR(100),
                    attribute_id INTEGER,
                    FOREIGN KEY (attribute_id) REFERENCES product_attribute(id)
                );

                CREATE TABLE product_template_attribute_value (
                    id INTEGER PRIMARY KEY,
                    product_tmpl_id INTEGER,
                    product_attribute_value_id,
                    price_extra NUMERIC,
                    FOREIGN KEY (product_tmpl_id) REFERENCES product_template(id),
                    FOREIGN KEY (product_attribute_value_id) REFERENCES product_attribute_value(id)
                );

                CREATE TABLE product_product (
                    id INTEGER PRIMARY KEY,
                    product_tmpl_id INTEGER,
                    FOREIGN KEY (product_tmpl_id) REFERENCES product_template(id)
                );

                CREATE TABLE res_partner (
                    id INTEGER PRIMARY KEY,
                    name VARCHAR(100),
                    email VARCHAR(100)
                );

                CREATE TABLE sale_order (
                    id INTEGER PRIMARY KEY,
                    name VARCHAR(100),
                    date_order timestamp without time zone,
                    partner_id INTEGER,
                    amount_untaxed NUMERIC,
                    amount_total NUMERIC,
                    FOREIGN KEY (partner_id) REFERENCES res_partner(id)
                );

                CREATE TABLE sale_order_line (
                    id INTEGER PRIMARY KEY,
                    order_id INTEGER,
                    name VARCHAR(100),
                    price_unit NUMERIC,
                    price_subtotal NUMERIC,
                    price_total NUMERIC,
                    product_id INTEGER, 
                    product_uom_qty NUMERIC,
                    FOREIGN KEY (order_id) REFERENCES sale_order(id),
                    FOREIGN KEY (product_id) REFERENCES product_product(id)
                );
            ```
        """

    def convert_to_sql(self, text_for_query: str):  
        completion = openai.Completion.create(engine="text-davinci-003",
            prompt="Eres un asistente SQL de un módulo del ERP Odoo. Tú misión es convertir una consulta en lenguaje natural a una query de sql que devuelva los datos que te piden. Solo debes contestar con la consulta SQL, nada más. Conviérteme a una query sql la siguiente petición:" + text_for_query + "\n, Dentro de una base de datos postgres creada con el siguiente script: " + self._creation_db_script + "\n Quiero que utilices la sintaxis 'table_name.attribute' para las columnas del select.",
            max_tokens=100)
        return completion.choices[0].text