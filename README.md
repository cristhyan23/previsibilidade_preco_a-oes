# Previsão de Preços de Ações usando LSTM

Este projeto utiliza redes Long Short-Term Memory (LSTM) para prever os preços das ações para a próxima semana. O modelo LSTM é treinado com dados históricos de ações obtidos do Yahoo Finance.

## Visão Geral

O principal objetivo deste projeto é prever os preços futuros de várias ações ou criptomoedas treinando um modelo de aprendizado profundo usando LSTM. Os preços previstos são então analisados para fornecer insights sobre tendências de crescimento potenciais.

## Dependências

Este projeto requer as seguintes bibliotecas Python:

- `yfinance`: Para recuperar dados de ações do Yahoo Finance.
- `numpy`: Para operações numéricas.
- `tensorflow` e `keras`: Para implementar o modelo LSTM.
- `pandas`: Para manipulação e análise de dados.
- `scikit-learn`: Para pré-processamento de dados e avaliação do desempenho do modelo.

## Estrutura do Projeto

O projeto consiste nos seguintes componentes:

1. **Recuperação de Dados**: Dados de ações e criptomoedas são buscados no Yahoo Finance para um período especificado.
2. **Preparação de Dados**: Os dados obtidos são pré-processados e preparados para serem inseridos no modelo LSTM.
3. **Treinamento do Modelo**: Um modelo LSTM sequencial é definido e treinado usando os dados preparados.
4. **Previsão**: O modelo treinado é utilizado para prever os preços das ações para a próxima semana.
5. **Análise**: Os preços previstos são analisados para determinar as tendências de crescimento e a precisão do modelo.
6. **Saída**: Os resultados, incluindo previsões e análises, são salvos em um arquivo Excel.

## Uso

Para utilizar este projeto:

1. Garanta que todas as dependências estejam instaladas (`yfinance`, `numpy`, `tensorflow`, `keras`, `pandas`, `scikit-learn`).
2. Execute o script `stock_price_prediction.py`.
3. O script irá gerar previsões para as ações ou criptomoedas especificadas e salvar os resultados em um arquivo Excel com análises sobre a diferença na previsão em relação ao último preço e a diferença em relação a cada previsão 7 dias à frente, nomeado `predictions.xlsx`.

## Observação

- Este projeto é destinado apenas para fins educacionais e experimentais.
- A previsão de preços de ações e Cripto é inerentemente incerta e influenciada por vários fatores. Os resultados podem nem sempre ser precisos ou confiáveis.
- É recomendável consultar especialistas financeiros antes de tomar qualquer decisão de investimento com base nas previsões geradas por este projeto.
- Ao selecionar Cripto, a moeda será mostrada em USD; se selecionar ações, a moeda será em BRL.
- A variavel na classe AnalisisStock chamada filter_params contem os parametros de caliberam para as previsões de 7 dias à frente focando otimizar e encontrar resultados mais alinhados a curva, os mesmos devem ser ajustados de acordo com a análise feita
