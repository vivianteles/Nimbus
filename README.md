# Nimbus: Estratégia Quantitativa (Clima + Sazonalidade)
Este repositório contém o código-fonte do Nimbus, um robô de investimento quantitativo desenvolvido para o Desafio Quant AI 2025 do Itaú Asset Management .

O projeto utiliza Machine Learning (XGBoost) e dados climáticos alternativos (Meteostat) para prever e operar o ETF de Milho (CORN).




💡 Tese de Investimento:
A tese central é que o impacto dos dados climáticos (chuva, temperatura) nos preços das commodities agrícolas não é linear, mas sim condicional à sazonalidade. Uma seca em Iowa em maio (plantio) tem um impacto muito maior no preço do milho do que uma seca em dezembro (entressafra).


O Nimbus explora essa ineficiência usando um modelo de Machine Learning (max_depth=5) capaz de entender a complexa interação entre clima e o mês ('month')  para antecipar movimentos de preço.



⚙️ Framework Técnico:
A estratégia é executada mensalmente e segue um fluxo rigoroso:


Coleta de Dados: O robô busca dados climáticos (TAVG, PRCP) do Meteostat para regiões-chave (Iowa-EUA e Mato Grosso-BR) e dados de preço do CORN (yfinance).




Engenharia de Features: Os dados brutos são transformados em features (lags de 1, 3, 6 meses; médias móveis; anomalias) e a feature de sazonalidade (month) é extraída.


Modelo Preditivo: Um classificador XGBoost (max_depth=5, n_estimators=200, learning_rate=0.05) é treinado (usando TimeSeriesSplit com n_splits=5)  para calcular a probabilidade de alta do CORN no próximo mês.



Estratégia "Sniper": Para filtrar o ruído do mercado, o robô só opera em sinais de alta convicção.


Sinal de Compra: Probabilidade de Alta > 70% 


Sinal de Venda: Probabilidade de Alta < 30% 


Sinal de Manter: (Sinal 0) Probabilidade entre 30% e 70%.


🛠️ Stack Tecnológica:
Python

Pandas (Manipulação de dados)

XGBoost (Modelagem)

Meteostat (Dados climáticos)

yfinance (Dados de mercado)

QuantStats (Análise de Backtest)

Seaborn / Matplotlib (Visualização)
