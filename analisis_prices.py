# -*- coding: utf-8 -*-
import pandas as pd
import yfinance as yf
from stockslist import Stocklist

class AddPriceCompare(Stocklist):
    def __init__(self):
        super().__init__()
        
    #valida se o arquivo predictions existe caso não cria um em branco
    def read_file(self):
        try:
            df = pd.read_excel("./predictions.xlsx")
            return df
        except FileNotFoundError:
            data_file = [{
                        "stocks": '', 
                        "predictions_1": '',
                        "growth_index": '',
                        "analisis r2":'',
                        "ultimo_preco":''
                    }]
            df = pd.DataFrame(data_file)
            return df
      
#função responsavel por capturar as ações levantar o ultimo preço e salvar no arquivo    
    def add_last_price(self):
        df = self.read_file()        
        stock_info = df["stocks"]  
        print("starting data analisis...")
        for stock in stock_info:
            try:
                acao = yf.Ticker(stock)
                historico_precos = acao.history(period="1d")
                # Extraia o último preço de fechamento
                print("getting the last price of the stock....")
                ultimo_preco_fechamento = historico_precos["Close"].iloc[-1]
                # Atualiza o valor no DataFrame
                df.loc[df['stocks'] == stock, 'ultimo_preco'] = ultimo_preco_fechamento
            except Exception as e:
                print(f'Error Ocur: {e}')
                continue
        print("updating the excel file....")
        df.to_excel('./predictions.xlsx',index=False, engine='xlsxwriter')

    # Função para analisar os dados de último preço x a primeira previsão e apontar projeção de queda e aumento
    def add_diferences_prediciton_add_last_price(self,days_ahead):
        df = self.read_file()
        print("generating analisis x last price")
        stock_info = df["stocks"] 
        for stock in stock_info:
            #adiciona a previsibilidade x o último preco
            prediction_1 = df['predictions_1']
            last_price = df['ultimo_preco']
            price_differences_1 = (prediction_1 / last_price - 1)
            column_name = 'delta_ultimo_preco_vs_1_prediction'
            df.loc[df['stocks'] == stock, column_name] = price_differences_1

            #loop para adicionar a previsibilidade indepente do volume de dias para frente
            for days in range(1,days_ahead):
                prediction_data_current = df[f'predictions_{days}']
                prediction_data_next = df[f'predictions_{days+1}']
                price_differences = (prediction_data_next / prediction_data_current - 1)
                column_name = f'delta_{days}_prediction_vs_{days+1}_prediction'
                df.loc[df['stocks'] == stock, column_name] = price_differences

        #salva o novo dataframe com os valores de previsões
        print("final file updating....")
        df = df.drop([f'predictions_{days_ahead}',f'delta_{days_ahead-1}_prediction_vs_{days_ahead}_prediction'],axis=1)
        df.to_excel('./predictions.xlsx',index=False, engine='xlsxwriter')


    