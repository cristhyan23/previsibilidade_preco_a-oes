# -*- coding: utf-8 -*-
import pandas as pd
import yfinance as yf
from stockslist import Stocklist

class AddPriceCompare(Stocklist):
    def __init__(self):
        super().__init__()
        self.df = pd.read_excel("./predictions.xlsx") 
#função responsavel por capturar as ações levantar o ultimo preço e salvar no arquivo    
    def add_last_price(self):
        stock_info = self.df["stocks"]  
        print("starting data analisis...")
        for stock in stock_info:
            try:
                acao = yf.Ticker(stock)
                historico_precos = acao.history(period="1d")
                # Extraia o último preço de fechamento
                print("getting the last price of the stock....")
                ultimo_preco_fechamento = historico_precos["Close"].iloc[-1]
                # Atualiza o valor no DataFrame
                self.df.loc[self.df['stocks'] == stock, 'ultimo_preco'] = ultimo_preco_fechamento
            except Exception as e:
                print(f'Error Ocur: {e}')
                continue
        print("updating the excel file....")
        self.df.to_excel('./predictions.xlsx',index=False, engine='xlsxwriter')

    # Função para analisar os dados de último preço x a primeira previsão e apontar projeção de queda e aumento
    def add_diferences_prediciton_add_last_price(self):
        print("generating analisis x last price")
        stock_info = self.df["stocks"] 
        for stock in stock_info:
            #analise comparativa da primeira previsão x o último preço da bolsa
            prediction_data_1 = self.df['predictions_1']
            last_price = self.df['ultimo_preco']
            price_diferences_1 = (prediction_data_1 / last_price - 1) *100
            self.df.loc[self.df['stocks'] == stock, 'delta_ultimo_preco_vs_1_prediction'] = price_diferences_1
            #variavel da primeira previsão para a segunda
            prediction_data_2 = self.df['predictions_2']
            price_diferences_2 = (prediction_data_2 / prediction_data_1 - 1) *100
            self.df.loc[self.df['stocks'] == stock, 'delta_1_prediction_vs_2_prediction'] = price_diferences_2
            #variavel da segunda previsão para a terceira
            prediction_data_3 = self.df['predictions_3']
            price_diferences_3 = (prediction_data_3 / prediction_data_2 - 1) *100
            self.df.loc[self.df['stocks'] == stock, 'delta_2_prediction_vs_3_prediction'] = price_diferences_3
            #variavel da terceira previsão para a quarta
            prediction_data_4 = self.df['predictions_4']
            price_diferences_4 = (prediction_data_4 / prediction_data_3 - 1) *100
            self.df.loc[self.df['stocks'] == stock, 'delta_3_prediction_vs_4_prediction'] = price_diferences_4
            #variavel da quarta previsão para a quinta
            prediction_data_5 = self.df['predictions_5']
            price_diferences_5 = (prediction_data_5 / prediction_data_4 - 1) *100
            self.df.loc[self.df['stocks'] == stock, 'delta_4_prediction_vs_5_prediction'] = price_diferences_5
            #variavel da quinta previsão para a sexta
            prediction_data_6 = self.df['predictions_6']
            price_diferences_6 = (prediction_data_6 / prediction_data_5 - 1) *100
            self.df.loc[self.df['stocks'] == stock, 'delta_5_prediction_vs_6_prediction'] = price_diferences_6
            #variavel da sexta previsão para a sétima
            prediction_data_7 = self.df['predictions_7']
            price_diferences_7 = (prediction_data_7 / prediction_data_6 - 1) *100
            self.df.loc[self.df['stocks'] == stock, 'delta_6_prediction_vs_7_prediction'] = price_diferences_7
        #salva o novo dataframe com os valores de previsões
        print("final file updating....")
        self.df.to_excel('./predictions.xlsx',index=False, engine='xlsxwriter')


    