import os

import mysql.connector
import openai
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Dash Madamy Ubajara",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def conexãoUbj():
    ######################### CONEXÃO LOJA UBAJARA ########################################
    global conexao
    conexao = mysql.connector.connect(
        host="srv496.hstgr.io",
        database="u771541202_db_madamy",
        user="u771541202_madamy",
        password="visioWebsolucao2",
    )

    if conexao.is_connected():
        print("conexão ok")

        querycaixa = "SELECT * FROM caixas"
        querycaixaItens = "SELECT * FROM caixa_items"
        querycargos = "SELECT * FROM cargos"
        queryclientes = "SELECT * FROM clientes"
        querycolaboradores = "SELECT * FROM colaboradors"
        querycomandas = "SELECT * FROM comandas"
        queryempresas = "SELECT * FROM empresas"
        queryestoques = "SELECT * FROM estoques"
        queryforma_pagamentos = "SELECT * FROM forma_pagamentos"
        queryfornecedores = "SELECT * FROM fornecedors"
        querygrupo_produtos = "SELECT * FROM grupo_produtos"
        queryitem_comandas = "SELECT * FROM item_comandas"
        queryitem_estoques = "SELECT * FROM item_estoques"
        querylojas = "SELECT * FROM lojas"
        queryprodutos = "SELECT * FROM produtos"
        queryproduto_precos = "SELECT * FROM produto_precos"
        querypromocaos = "SELECT * FROM promocaos"

        global dfCaixa, dfCaixaItens, dfCargos, dfClientes, dfColaboradores, dfComandas, dfEmpresas, dfEstoques, dfFormaPagamento, dfFornecedores, dfGrupoProdutos, dfItemComandas, dfItemEstoque, dfLojas, dfProdutos, dfProdutoPreco, dfPromocoes

        dfCaixa = pd.read_sql(querycaixa, conexao)
        dfCaixaItens = pd.read_sql(querycaixaItens, conexao)
        dfCargos = pd.read_sql(querycargos, conexao)
        dfClientes = pd.read_sql(queryclientes, conexao)
        dfColaboradores = pd.read_sql(querycolaboradores, conexao)
        dfComandas = pd.read_sql(querycomandas, conexao)
        dfEmpresas = pd.read_sql(queryempresas, conexao)
        dfEstoques = pd.read_sql(queryestoques, conexao)
        dfFormaPagamento = pd.read_sql(queryforma_pagamentos, conexao)
        dfFornecedores = pd.read_sql(queryfornecedores, conexao)
        dfGrupoProdutos = pd.read_sql(querygrupo_produtos, conexao)
        dfItemComandas = pd.read_sql(queryitem_comandas, conexao)
        dfItemEstoque = pd.read_sql(queryitem_estoques, conexao)
        dfLojas = pd.read_sql(querylojas, conexao)
        dfProdutos = pd.read_sql(queryprodutos, conexao)
        dfProdutoPreco = pd.read_sql(queryproduto_precos, conexao)
        dfPromocoes = pd.read_sql(querypromocaos, conexao)
    return (
        dfCaixa,
        dfCaixaItens,
        dfCargos,
        dfClientes,
        dfColaboradores,
        dfComandas,
        dfEmpresas,
        dfEstoques,
        dfFormaPagamento,
        dfFornecedores,
        dfGrupoProdutos,
        dfItemComandas,
        dfItemEstoque,
        dfLojas,
        dfProdutos,
        dfProdutoPreco,
        dfPromocoes,
    )


def plot_total_vendas_por_mes(dfCaixaItens, year):
    # Filtrar o DataFrame com base no ano selecionado
    df_filtrado = dfCaixaItens[dfCaixaItens["Year"].isin(year)]

    # Agrupar as vendas por mês e calcular o valor total de vendas
    vendas_por_mes = df_filtrado.groupby("Month")["valor"].sum()

    # Gráfico de linhas com o total de vendas por mês
    fig = go.Figure(
        data=[go.Scatter(x=vendas_por_mes.index, y=vendas_por_mes.values, mode="lines")]
    )

    fig.update_layout(
        title="Total de Vendas por Mês",
        xaxis_title="Mês",
        yaxis_title="Valor Total de Vendas",
    )

    return fig


def plot_total_vendas_por_Ano(dfCaixaItens, year):
    # Filtrar o DataFrame com base no ano selecionado na sidebar
    df_filtrado = dfCaixaItens[dfCaixaItens["Year"].isin(year)]

    # Agrupar as vendas por mês e calcular o valor total de vendas
    vendas_por_ano = df_filtrado.groupby("Year")["valor"].sum()

    # Gráfico de linhas com o total de vendas por mês
    fig = go.Figure(data=[go.Line(x=vendas_por_ano.index, y=vendas_por_ano.values)])

    fig.update_layout(
        title="Total de Vendas por Ano",
        xaxis_title="Ano",
        yaxis_title="Valor Total de Vendas",
    )

    return fig


def plot_formas_pagamento(dfCaixaItens, dfFormaPagamento, year, month):
    # Filtrar o DataFrame com base nos valores selecionados na sidebar
    df_filtrado = dfCaixaItens.loc[
        (dfCaixaItens["Year"].isin(year)) & (dfCaixaItens["Month"].isin(month))
    ]

    # Mapear os IDs de forma de pagamento para as descrições correspondentes
    df_filtrado_com_descricao = df_filtrado.merge(
        dfFormaPagamento, left_on="forma_pagamento_id", right_on="id"
    )

    # Contagem das formas de pagamento
    formas_pagamento = df_filtrado_com_descricao["forma_pagamento"].value_counts()

    # Gráfico de barras com as formas de pagamento
    fig = go.Figure(data=[go.Bar(x=formas_pagamento.index, y=formas_pagamento.values)])

    fig.update_layout(
        title="Formas de Pagamento",
        xaxis_title="Forma de Pagamento",
        yaxis_title="Quantidade",
    )

    return fig


def plot_quantidade_por_grupo_produto(dfGrupoProdutos, dfProdutos):
    # Merge entre os DataFrames de grupo de produtos e produtos
    df_merge = dfProdutos.merge(
        dfGrupoProdutos, left_on="grupo_produto_id", right_on="id"
    )

    # Contagem da quantidade de produtos por grupo de produtos
    quantidade_por_grupo_produto = df_merge["grupo"].value_counts()

    # Gráfico de barras com a quantidade de produtos por grupo de produtos
    fig = go.Figure(
        data=[
            go.Bar(
                x=quantidade_por_grupo_produto.index,
                y=quantidade_por_grupo_produto.values,
            )
        ]
    )

    fig.update_layout(
        title="Quantidade de Produtos por Grupo de Produtos",
        xaxis_title="Grupo de Produtos",
        yaxis_title="Quantidade de Produtos",
    )

    return fig


def plot_hist_vendas_por_dia_semana(dfCaixaItens, year, month):
    # Filtrar o DataFrame dfCaixaItens com base nas opções selecionadas na sidebar
    df_filtrado = dfCaixaItens[
        (dfCaixaItens["Year"].isin(year)) & (dfCaixaItens["Month"].isin(month))
    ]

    # Criar uma nova coluna "DiaSemana" com base na coluna "created_at"
    df_filtrado["DiaSemana"] = df_filtrado["created_at"].apply(lambda x: x.weekday())

    # Mapear o número do dia da semana para o nome do dia
    dias_semana = [
        "Segunda",
        "Terça",
        "Quarta",
        "Quinta",
        "Sexta",
        "Sábado",
        "Domingo",
    ]
    df_filtrado["DiaSemana"] = df_filtrado["DiaSemana"].map(lambda x: dias_semana[x])

    # Agrupar por "DiaSemana" e contar as vendas
    vendas_por_dia = (
        df_filtrado[df_filtrado["descricao"] == "venda"].groupby("DiaSemana").size()
    )

    # Criar o gráfico de barras
    fig = px.bar(
        vendas_por_dia,
        x=vendas_por_dia.index,
        y=vendas_por_dia.values,
        labels={"x": "Dia da Semana", "y": "Quantidade de Vendas"},
    )

    # Personalizar o layout do gráfico
    fig.update_layout(
        title="Histórico de Vendas por Dia da Semana",
        xaxis_tickmode="linear",
        yaxis=dict(title="Quantidade de Vendas"),
    )

    # Exibir o gráfico# Filtrar o DataFrame dfCaixaItens com base nas opções selecionadas na sidebar
    df_filtrado = dfCaixaItens[
        (dfCaixaItens["Year"].isin(year)) & (dfCaixaItens["Month"].isin(month))
    ]

    # Criar uma nova coluna "DiaSemana" com base na coluna "created_at"
    df_filtrado["DiaSemana"] = df_filtrado["created_at"].apply(lambda x: x.weekday())

    # Mapear o número do dia da semana para o nome do dia
    dias_semana = [
        "Segunda",
        "Terça",
        "Quarta",
        "Quinta",
        "Sexta",
        "Sábado",
        "Domingo",
    ]
    df_filtrado["DiaSemana"] = df_filtrado["DiaSemana"].map(lambda x: dias_semana[x])

    # Agrupar por "DiaSemana" e contar as vendas
    vendas_por_dia = (
        df_filtrado[df_filtrado["descricao"] == "venda"].groupby("DiaSemana").size()
    )

    # Criar o gráfico de barras
    fig = px.bar(
        vendas_por_dia,
        x=vendas_por_dia.index,
        y=vendas_por_dia.values,
        labels={"x": "Dia da Semana", "y": "Quantidade de Vendas"},
    )

    # Personalizar o layout do gráfico
    fig.update_layout(
        title="Histórico de Vendas por Dia da Semana",
        xaxis_tickmode="linear",
        yaxis=dict(title="Quantidade de Vendas"),
    )

    # Exibir o gráfico
    return fig


def plot_vendas_por_forma_pagamento(dfCaixaItens, dfFormaPagamento, year, month):
    # Filtrar o DataFrame dfCaixaItens com base nos valores de year e month
    df_filtered = dfCaixaItens[
        (dfCaixaItens["Year"].isin(year)) & (dfCaixaItens["Month"].isin(month))
    ]

    # Juntar os DataFrames de comandas e forma de pagamento
    df_joined = df_filtered.merge(
        dfFormaPagamento, left_on="forma_pagamento_id", right_on="id"
    )

    # Calcular o total de vendas por forma de pagamento
    vendas_por_forma_pagamento = (
        df_joined.groupby("forma_pagamento")["valor"].sum().reset_index()
    )

    # Criar o gráfico de pizza
    fig = px.pie(vendas_por_forma_pagamento, values="valor", names="forma_pagamento")

    fig.update_layout(title="Vendas por Forma de Pagamento")

    return fig


def criarDash():
    st.header("🎉 Bem-vindo ao Dashboard da Madamy Acessórios Ubajara! 🛍️")
    st.markdown("---")

    ################### CRIAR COLUNAS DE MES E ANO DE ALGUNS DATAFRAME #######################
    dfCaixaItens["Month"] = dfCaixaItens["created_at"].apply(lambda x: x.month)
    dfCaixaItens["Year"] = dfCaixaItens["created_at"].apply(lambda x: x.year)

    ##################  SIDEBAR   ######################################################

    valores_unicos_year = dfCaixaItens["Year"].unique()
    default_values_year = list(valores_unicos_year)
    year = st.sidebar.multiselect(
        key=1,
        label="Ano",
        options=dfCaixaItens["Year"].unique(),
        default=default_values_year,
    )

    valores_unicos_moth = dfCaixaItens["Month"].unique()
    default_values_moth = list(valores_unicos_moth)
    month = st.sidebar.multiselect(
        key=2,
        label="Mês",
        options=dfCaixaItens["Month"].unique(),
        default=default_values_moth,
    )

    #####################  TELA PRINCIPAL ###########################

    # Filtrar o DataFrame dfCaixaItens com base nas opções selecionadas na sidebar
    df_filtrado = dfCaixaItens[
        (dfCaixaItens["Year"].isin(year)) & (dfCaixaItens["Month"].isin(month))
    ]
    # 1 - Quantidade de dfCaixaItens["descricao"] == "venda"
    quantidade_vendas = len(df_filtrado[df_filtrado["descricao"] == "venda"])

    # 2 - Valor Total de Entrada de Todos as Vendas de entrada
    valor_total_entradas = df_filtrado[df_filtrado["descricao"] == "venda"][
        "valor"
    ].sum()
    valor_total_entradas_Format = f"R$ {valor_total_entradas:,.2f}".replace(",", ".")

    # 3 - Valor Total de Desconto
    valor_total_desconto = df_filtrado["valor_desconto"].sum()
    valor_total_desconto_Format = f"R$ {valor_total_desconto:,.2f}".replace(",", ".")

    # 4 - Porcentagem de Vendas dos últimos 7 dias
    hoje = pd.Timestamp.today().date()
    data_7_dias_atras = hoje - pd.DateOffset(days=7)
    df_ultimos_7_dias = df_filtrado[
        (df_filtrado["created_at"].dt.date >= pd.Timestamp(data_7_dias_atras).date())
        & (df_filtrado["created_at"].dt.date <= pd.Timestamp(hoje).date())
    ]
    quantidade_vendas_7_dias = len(
        df_ultimos_7_dias[df_ultimos_7_dias["descricao"] == "venda"]
    )
    porcentagem_vendas_7_dias = quantidade_vendas_7_dias / quantidade_vendas * 100

    # 6 - Valor Total de Entradas dos últimos 7 dias
    valor_total_entradas_7_dias = df_ultimos_7_dias[
        df_ultimos_7_dias["descricao"] == "venda"
    ]["valor"].sum()
    porcentagem_valor_entradas_7_dias = (
        valor_total_entradas_7_dias / valor_total_entradas * 100
    )

    # 5 - Porcentagem de Vendas do dia atual
    df_dia_atual = df_filtrado[
        df_filtrado["created_at"].dt.date == pd.Timestamp(hoje).date()
    ]
    quantidade_vendas_dia_atual = len(
        df_dia_atual[df_dia_atual["descricao"] == "venda"]
    )
    porcentagem_vendas_dia_atual = quantidade_vendas_dia_atual / quantidade_vendas * 100
    # st.text(quantidade_vendas_dia_atual)
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    col1.metric(
        "🤝 Quantidade de Vendas e Vendas últimos 7 dias (%)",
        quantidade_vendas,
        f"{porcentagem_vendas_7_dias:.2f}%",
    )
    col2.metric(
        "Total 💵 Pedidos",
        valor_total_entradas_Format,
        f"{porcentagem_valor_entradas_7_dias:.2f}%",
    )
    col3.metric("Valor Desconto 📊", valor_total_desconto_Format)
    col4.metric("Vendas Dia (%)", porcentagem_vendas_dia_atual)

    col1_, col2_ = st.columns(2)

    with col1_:
        # Gráfico de formas de pagamento
        fig_formas_pagamento = plot_formas_pagamento(
            dfCaixaItens, dfFormaPagamento, year, month
        )
        st.plotly_chart(fig_formas_pagamento)

    with col2_:
        # Gráfico de vendas por forma de pagamento
        fig_vendas_por_forma_pagamento = plot_vendas_por_forma_pagamento(
            dfCaixaItens, dfFormaPagamento, year, month
        )
        st.plotly_chart(fig_vendas_por_forma_pagamento)

    with col1_:
        fig_total_vendas = plot_total_vendas_por_mes(dfCaixaItens, year)
        st.plotly_chart(fig_total_vendas)

    with col2_:
        fig_total_vendas = plot_total_vendas_por_Ano(dfCaixaItens, year)
        st.plotly_chart(fig_total_vendas)

    with col1_:
        fig_plot_hist_vendas_por_dia_semana = plot_hist_vendas_por_dia_semana(
            dfCaixaItens, year, month
        )
        st.plotly_chart(fig_plot_hist_vendas_por_dia_semana)

    with col2_:
        fig_quantidade_por_grupo_produto = plot_quantidade_por_grupo_produto(
            dfGrupoProdutos, dfProdutos
        )
        st.plotly_chart(fig_quantidade_por_grupo_produto)

    openai.api_key = os.getenv("OPENAI_API_KEY")
    chatgpt = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": f"""A loja Madamy Acessorios, vende acesorios femininos como brincos e colares, as informações de {quantidade_vendas} representa a quantidae de pedidos, o {valor_total_entradas_Format} repreenta o valor em R$ de vendas, o {valor_total_desconto_Format} é o desconto aplicado em vendas, o Dataset {fig_formas_pagamento} é o grafico que corresponde a forma de pagamento mais usada nas comandas, o dataset {fig_vendas_por_forma_pagamento} é o grafico que corresponde os valores pagos por cadas forma de pagamento, o dataset {fig_total_vendas} é o grafico que correponde ao total de vendas em cada mês, o dataset {fig_plot_hist_vendas_por_dia_semana} é o grafico que corresponde o historico de vendas por dia da semana. Me informe 5 insight para minha loja """,
            },
        ],
    )
    insights = chatgpt["choices"][0]["message"]["content"]
    if st.button("Analisar"):
        st.write(insights)


(
    dfCaixa,
    dfCaixaItens,
    dfCargos,
    dfClientes,
    dfColaboradores,
    dfComandas,
    dfEmpresas,
    dfEstoques,
    dfFormaPagamento,
    dfFornecedores,
    dfGrupoProdutos,
    dfItemComandas,
    dfItemEstoque,
    dfLojas,
    dfProdutos,
    dfProdutoPreco,
    dfPromocoes,
) = conexãoUbj()
criarDash()
