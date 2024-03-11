# -*- coding: utf-8 -*-
import yfinance as yf
import numpy as np
from datetime import date, timedelta
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout,InputLayer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.metrics import r2_score
import unicodedata
import sys
import codecs
from analisis_prices import AddPriceCompare

class AnalisisStock(AddPriceCompare):
    def __init__(self):
        super().__init__()
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data_list = []
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
        self.epochs_database = 30
        self.dias_base_estudo = 5000
        self.look_back = self.dias_base_estudo *0.8
        self.filter_params = {
    'scale_factor': 0.1,  # Fator de escala para o segundo dia
    'max_percent_change': 0.2,  # Variação percentual máxima permitida entre os preços previstos
    'adjustment_factor': 0.05  # Fator de ajuste para a previsão anterior em caso de variação percentual alta
}
    # Função para substituir ou remover caracteres não suportados
    def sanitize_text(self, text):
        sanitized_text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
        return sanitized_text
    
    # Função para imprimir texto codificado corretamente
    def print_encoded(self, text):
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())  # UTF-8 output
        print(text)  

# Função para preparar dados para LSTM
    def prepare_data(self,data, look_back=1):
        X, Y = [], []
        for i in range(len(data)-look_back-1):
            X.append(data[i:(i+look_back), :])
            Y.append(data[i + look_back, 3])  # 3 corresponde ao índice do preço de fechamento (Close)
        return np.array(X), np.array(Y)
# A suavização exponencial atribui pesos decrescentes aos pontos de dados mais antigos, dando mais importância aos pontos de dados recentes. Isso é útil quando há uma tendência em mudança na série temporal.
    def smooth_with_exponential_smoothing(self,data, alpha):
        smoothed_data = [data[0]]
        for i in range(1, len(data)):
            smoothed_value = alpha * data[i] + (1 - alpha) * smoothed_data[-1]
            smoothed_data.append(smoothed_value)
        return smoothed_data


    # Função para fazer previsões
    def predict_next_week(self, model, data, days_ahead=7, filter_params=None):
        prediction = []
        last_sequence = data[-days_ahead:, :]
        for day in range(days_ahead):
            # Predição do próximo dia
            scaled_prediction = model.predict(np.array([last_sequence]))[0][0]
            
            # Inversão da escala para obter o valor não normalizado
            unscaled_prediction = self.scaler.inverse_transform([[0, 0, 0, scaled_prediction]])[0][3]
            prediction.append(unscaled_prediction)
            # Atualizando a sequência de entrada para o próximo dia
            last_sequence = np.roll(last_sequence, -1, axis=0)
            last_sequence[-1, 3] = unscaled_prediction  # Substituir o último preço de fechamento pelo previsto

            # Aplicar fator de escala específico para o segundo dia
            if day >=1:
                smoothed_prediction = self.smooth_with_exponential_smoothing(prediction, alpha=0.2)
                prediction =smoothed_prediction
            
            # Filtragem mais detalhada do próximo dia
            if filter_params:
                # Verificar a variação percentual entre os preços previstos para os próximos dias
                if day >= 2 and len(prediction) >= 2:
                    percent_change = (prediction[day] - prediction[day - 1]) / prediction[day - 1] * 100
                    if percent_change > filter_params.get('max_percent_change'):
                        # Se a variação percentual for muito alta, ajustar a previsão anterior
                        prediction[day - 1] *= 1 + filter_params.get('adjustment_factor')
                        # Atualizar a sequência de entrada com a previsão ajustada
                        last_sequence[day - 1, 3] = prediction[day - 1]
            
        return prediction


    #função que prepara o treino e retorna o modelo
    def prepar_training_and_model(self,X,Y):
        # Dividindo os dados em treinamento e teste
        train_size = int(len(X) * 0.9)
        test_size = len(X) - train_size
        X_train, X_test = X[:train_size], X[train_size:]
        Y_train, Y_test = Y[:train_size], Y[train_size:]
        # Definindo arquitetura do modelo LSTM
        model = Sequential()
        model.add(InputLayer(shape=(self.look_back, 4)))
        model.add(LSTM(128, return_sequences=True))
        model.add(Dropout(0.2))  # Adicionando camada de dropout
        model.add(LSTM(64, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        # Compilando o modelo
        model.compile(optimizer='adam', loss='mean_squared_error')
        # Ajustando o modelo aos dados de treinamento
        model.fit(X_train, Y_train, batch_size=2, epochs=self.epochs_database, verbose=2)
        return model,X_train,Y_train
    
    #função que pepara os dados para LSTM
    def prepar_data_lstm(self,data_features):
            data_features_scaled = self.scaler.fit_transform(data_features)
            X, Y = self.prepare_data(data_features_scaled, self.look_back)
            return X,Y,data_features_scaled
    
    #Função para remover outliers do historio de preço de fechamento das ações
    def remove_outliers(self,data):
        # Calcular a média e o desvio padrão dos preços de fechamento
        mean = data['Close'].mean()
        std_dev = data['Close'].std()
        # Definir o limite para identificar outliers (por exemplo, 3 desvios padrão da média)
        outlier_limit = mean + 3 * std_dev
        # Remover outliers (valores de fechamento que excedem o limite)
        filtered_data = data[data['Close'] < outlier_limit]
        return filtered_data

    #Função que gera dados históricos das ações
    def get_stock(self,acoes):
        # Obtendo dados das ações
                today = date.today()
                d1 = today.strftime("%Y-%m-%d")
                end_date = d1
                d2 = date.today() - timedelta(days=self.dias_base_estudo)
                d2 = d2.strftime("%Y-%m-%d")
                start_date = d2

                stock_ticker = acoes
                data = yf.download(stock_ticker,
                                    start=start_date,
                                    end=end_date,
                                    progress=False)
                data["Date"] = data.index
                data = data[["Date", "Open", "High", "Low", "Close",
                            "Adj Close", "Volume"]]
                # Remover outliers dos dados de fechamento
                data = self.remove_outliers(data)

                data.reset_index(drop=True, inplace=True)
                return data

    #analise de precisão do modelo será feito por stock
    def analisis_results_r2(self,model,X_train,Y_train):
        # Predict on the training set
        y_train_pred = model.predict(X_train)
        # Calculate R^2 score
        r2 = r2_score(Y_train, y_train_pred)
        return r2*100

    #gera as previsões dos proximos 7 dias de cada ação salva na lista da classe stockslist   
    def generate_analisis(self):
        #captura a lista de ações
        list_acoes = self.get_list_stocks()
        #executa previsbildiade com o estudo de modelo em cada ação
        for acoes in list_acoes:
            
            acoes_cleaned = self.sanitize_text(acoes)
            print(f'iniciating analisis: {acoes_cleaned}')
            try:
                data = self.get_stock(acoes)
                # Convertendo dataframe para array numpy
                data_array = data.values
                # Preparando dados para LSTM
                data_features = data[['Open', 'High', 'Low', 'Close']].values
                X,Y,data_features_scaled = self.prepar_data_lstm(data_features)
                model,X_train,Y_train = self.prepar_training_and_model(X,Y)
                #analise de precisão do modelo quanto mais proximo de 100% melhor
                analisis_r2 = self.analisis_results_r2(model,X_train,Y_train)
                # Fazendo previsões para a próxima semana
                latest_data = data_features_scaled[-self.look_back:, :]
                predictions = self.predict_next_week(model, latest_data,days_ahead=7,filter_params=self.filter_params)
                growth_index = (predictions[-1] / predictions[0] - 1) * 100
               
                #guarda as previsões e projeções das ações
                print(f'saving data: {acoes_cleaned}')
                result_dict = {
                        "stocks": acoes_cleaned, 
                        "predictions_1": predictions[0],
                        "predictions_2": predictions[1],
                        "predictions_3": predictions[2],
                        "predictions_4": predictions[3],
                        "predictions_5": predictions[4],
                        "predictions_6": predictions[5],
                        "predictions_7": predictions[6],
                        "growth_index": growth_index,
                        "analisis r2":analisis_r2
                    }
                
                self.data_list.append(result_dict)

            #aponta qual erro que ocorreu na ação e salta para a próxima
            except Exception as e:
                    print(f"Error occurred for {self.sanitize_text(acoes)}: {e}")
                    #continue
        return self.data_list
    
    def save_file(self):
        # Converter o dicionário em um DataFrame do pandas
        df = pd.DataFrame(self.generate_analisis())
        # Salvar o DataFrame em um arquivo Excel
        df.to_excel('./predictions.xlsx',index=False,engine='xlsxwriter')
        #Executar analises de preços no arquivo
        buscar_ultimo_preco = self.add_last_price()
        adicionar_analises = self.add_diferences_prediciton_add_last_price()


if __name__ == "__main__":
     analiser = AnalisisStock()
     file = analiser.save_file()
    

