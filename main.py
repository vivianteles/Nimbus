import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from meteostat import Point, Daily
from datetime import datetime

# Modelagem e Validação
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier 

# NOVAS IMPORTAÇÕES PARA VISUALIZAÇÃO
from sklearn.metrics import confusion_matrix
import seaborn as sns 

# Backtest
import quantstats as qs

# Desativar avisos de chained_assignment do pandas
pd.options.mode.chained_assignment = None 

#################################################################
# ETAPA 1: CONFIGURAÇÕES GLOBAIS
#################################################################
print("Iniciando o Robô NIMBUS 2.2 (Teste com CORN)...")

# Ativo Alvo
TARGET_TICKER = "CORN"         
START_DATE = "2010-01-01"     

# Features Climáticas (Expansão Global) 
REGIAO_BR = {"nome": "Mato Grosso", "ponto": Point(-15.6, -56.1)}
REGIAO_US = {"nome": "Iowa (EUA)", "ponto": Point(41.8, -93.6)}

# Features Macro e Logísticas (Expansão de Dados) 
MACRO_TICKERS = ["USDBRL=X", "^BDI", "ZS=F", "ZC=F"]

# Configurações do Modelo 
N_SPLITS_CV = 5           
PROBA_THRESHOLD_BUY = 0.70    
PROBA_THRESHOLD_SELL = 0.30     

# ETAPA 2: FUNÇÕES DE COLETA DE DADOS

def fetch_climate_data(regiao, start_date, suffix):
    print(f"Baixando dados climáticos de: {regiao['nome']}...")
    end = datetime.today()
    clima = Daily(regiao['ponto'], datetime.strptime(start_date, "%Y-%m-%d"), end).fetch()
    
    if clima.empty:
        raise ValueError(f"Meteostat não retornou NENHUM dado para {regiao['nome']}.")

    clima = clima.ffill() 
    clima_mensal = clima.resample("ME").agg({
        "tavg": "mean", "tmax": "mean", "tmin": "mean", "prcp": "sum"
    })
    clima_mensal = clima_mensal.add_suffix(suffix)
    return clima_mensal

def fetch_target_data(ticker, start_date):
    print(f"Baixando dados do ativo alvo: {ticker}...")
    dados = yf.download(ticker, start=start_date, progress=False)
    if dados.empty:
        raise ValueError(f"Ativo alvo {ticker} não encontrado.")
    
    prices = dados["Close"].resample("ME").last()
    returns = prices.pct_change()
    
    df_target = pd.DataFrame({
        "price": prices.squeeze(),
        "return": returns.squeeze()
    })
    return df_target

def fetch_macro_data(tickers, start_date):
    print(f"Baixando dados macro e de logística: {tickers}...")
    dados = yf.download(tickers, start=start_date, progress=False)
    
    if dados.empty:
        print(f"Aviso: yfinance não retornou dados para os tickers macro: {tickers}")
        return pd.DataFrame()

    if len(tickers) > 1:
        prices_macro = dados["Close"].resample("ME").last().ffill()
    else:
        prices_macro = dados["Close"].resample("ME").last().ffill().to_frame(name=tickers[0])
    
    returns_macro = prices_macro.pct_change()
    returns_macro = returns_macro.rename(columns={c: f"{c}_ret" for c in tickers if c != "^BDI"})
    
    if "^BDI" in tickers:
        returns_macro["^BDI"] = prices_macro["^BDI"] 
        
    return returns_macro


# ETAPA 3: MONTAGEM DO DATASET PRINCIPAL

# 1. Coletar todos os dados
df_target = fetch_target_data(TARGET_TICKER, START_DATE)
df_clima_br = fetch_climate_data(REGIAO_BR, START_DATE, "_br")
df_clima_us = fetch_climate_data(REGIAO_US, START_DATE, "_us")
df_macro = fetch_macro_data(MACRO_TICKERS, START_DATE)

# 2. Juntar tudo
df_full = pd.concat([df_target, df_clima_br, df_clima_us, df_macro], axis=1)

# 3. Remover meses iniciais onde o ativo alvo ainda não existia
df_full = df_full.dropna(subset=['price'])

# 4. Pré-limpeza de colunas 100% NaN (o "Filtro Anti-^BDI")
cols_antes = set(df_full.columns)
df_full = df_full.dropna(axis=1, how='all') 
cols_depois = set(df_full.columns)
cols_removidas = cols_antes - cols_depois
if cols_removidas:
    print(f"\nAviso: As seguintes colunas foram removidas por conterem 100% NaN: {cols_removidas}")

print("\nDados brutos coletados e unidos (pós-limpeza):")
print(df_full.tail())

# ETAPA 4: ENGENHARIA DE FEATURES 

print("\nIniciando engenharia de features (lags, rolling, diffs)...")

def create_features(df):
    
    df['month'] = df.index.month
    feature_names = ['month']
    
    
    base_features = [col for col in df.columns if col not in ['price', 'return']]
    
    
    for col in base_features:
        if col not in df.columns: continue 
            
        feature_names.append(col)
        
        for lag in [1, 3, 6]:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
            feature_names.append(f'{col}_lag{lag}')
            
        df[f'{col}_roll3'] = df[col].rolling(3).mean()
        feature_names.append(f'{col}_roll3')

    if 'tavg_br' in df.columns and 'tavg_us' in df.columns:
        df['tavg_diff_br_us'] = df['tavg_br'] - df['tavg_us']
        feature_names.append('tavg_diff_br_us')
    if 'prcp_br' in df.columns and 'prcp_us' in df.columns:
        df['prcp_diff_br_us'] = df['prcp_br'] - df['prcp_us']
        feature_names.append('prcp_diff_br_us')
    
    for col in [c for c in base_features if ('tavg' in c or 'prcp' in c) and c in df.columns]:
        df[f'{col}_anom'] = df[col] - df[col].rolling(6, min_periods=6).mean()
        feature_names.append(f'{col}_anom')

    return df, list(set(feature_names))

df_featured, feature_names = create_features(df_full.copy())

# ETAPA 5: DEFINIÇÃO DO ALVO E LIMPEZA FINAL

# <-- VOLTAMOS À LÓGICA BINÁRIA (0 ou 1)
df_featured["target_return"] = df_featured["return"].shift(-1)
df_featured["target_class"] = np.where(df_featured["target_return"] > 0, 1, 0)

df_final = df_featured.dropna(subset=feature_names + ['target_class'])

if df_final.empty:
    raise ValueError("DataFrame ainda vazio. Verifique se TODOS os tickers (CORN, Clima, Macro) estão funcionando e têm dados sobrepostos.")

X = df_final[feature_names]
y = df_final["target_class"] 

print(f"\nDataset final pronto para treino. Features: {len(feature_names)}, Amostras: {len(X)}")

# ETAPA 6: MACHINE LEARNING (XGBoost + TimeSeriesSplit)

print(f"Treinando modelo XGBoost com TimeSeriesSplit (n_splits={N_SPLITS_CV})...")

model = XGBClassifier(
    n_estimators=200, 
    max_depth=5,            
    learning_rate=0.05,
    objective='binary:logistic', 
    random_state=42, 
    subsample=0.8
)

tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV)


preds_proba, preds_class, actuals_class, actuals_return, months = [], [], [], [], []

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    model.fit(X_train, y_train)
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred_class = model.predict(X_test)
    
    preds_proba.extend(y_pred_proba)
    preds_class.extend(y_pred_class)
    actuals_class.extend(y_test)
    actuals_return.extend(df_final.loc[df_final.index[test_idx], "target_return"])
    months.extend(df_final.index[test_idx])

# ETAPA 6.5: (NOVO) ANÁLISE DE FEATURES

print("\nAnalisando a importância das features (do último fold)...")
try:
    # Criar um DataFrame de importância
    df_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_  
    }).sort_values(by='importance', ascending=False)

    
    print(df_importance.head(15))

    # Plotar
    plt.figure(figsize=(10, 8))
    top_15_features = df_importance.head(15)
    sns.barplot(
        x='importance', 
        y='feature', 
        data=top_15_features, 
        palette='viridis' 
    )
    plt.title('Top 15 Features Mais Importantes')
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Erro ao gerar gráfico de importância de features: {e}")

# ETAPA 7: AVALIAÇÃO DO MODELO (Classificação)

if not actuals_class:
    print(f"\nNenhuma previsão foi gerada. O dataset de {len(X)} amostras é muito pequeno para {N_SPLITS_CV} splits.")
else:
    acc = accuracy_score(actuals_class, preds_class)
    report = classification_report(actuals_class, preds_class, zero_division=0)
    
    print(f"\nResultados da Validação (Classificação):")
    print(f"Acurácia: {acc:.2%}")
    print(report)

    # ETAPA 7.5: (NOVO) PLOT DA MATRIZ DE CONFUSÃO
    
    print("Gerando Matriz de Confusão...")
    try:
        cm = confusion_matrix(actuals_class, preds_class)
        plt.figure(figsize=(7, 5)) # <-- Voltamos ao 2x2
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d',    
            cmap='Blues',
            
            xticklabels=['Previsto: Caiu (0)', 'Previsto: Subiu (1)'],
            yticklabels=['Real: Caiu (0)', 'Real: Subiu (1)']
        )
        plt.title(f'Matriz de Confusão (Acurácia: {acc:.2%})')
        plt.ylabel('Valor Real')
        plt.xlabel('Valor Previsto')
        plt.show()
    except Exception as e:
        print(f"Erro ao gerar Matriz de Confusão: {e}")

    # Criar DataFrame de resultados
    result = pd.DataFrame({
        "month": months,
        "pred_probability": preds_proba, # <-- pred_proba de volta
        "pred_class": preds_class,
        "real_class": actuals_class,
        "real_return": actuals_return
    })

    # ETAPA 8: GERAÇÃO DE SINAIS E RETORNO DA ESTRATÉGIA
    
    
    
    result["signal"] = np.where(result["pred_probability"] > PROBA_THRESHOLD_BUY, 1,
                                 np.where(result["pred_probability"] < PROBA_THRESHOLD_SELL, -1, 0))
    
    result["recomendacao"] = result["signal"].map({1: "Comprar", -1: "Vender", 0: "Manter"})
    result['strategy_return'] = result['signal'] * result['real_return']
    
    result.to_csv("nimbus_v2_output.csv", index=False)
    print("\n Resultados detalhados salvos em nimbus_v2_output.csv")
    print(result.tail())

    # ETAPA 9: GRÁFICO DE RETORNO ACUMULADO
    
    plt.figure(figsize=(12, 6))
    plt.plot(result["month"], result["strategy_return"].cumsum(), label="Nimbus (Estratégia)", marker='o', linestyle='--')
    plt.plot(result["month"], result["real_return"].cumsum(), label="Buy & Hold (CORN)", marker='.', linestyle='-') # <-- Label atualizada
    plt.title(f"Nimbus — Backtest (XGBoost) vs Buy & Hold")
    plt.ylabel("Retorno Acumulado")
    plt.xlabel("Mês (Período de Validação)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

   
    # ETAPA 10: BACKTEST FORMAL (Quantstats HTML)
    
    print("\nGerando relatório de backtest profissional com Quantstats...")
    
    # Adicionar o atributo 'name' para corrigir o AttributeError do quantstats
    strategy_returns_series = pd.Series(
        result['strategy_return'].values, 
        index=pd.to_datetime(result['month']),
        name="Nimbus 2.2 Strategy"
    ).fillna(0)
    
    # Adicionar o atributo 'name'
    benchmark_series = pd.Series(
        result['real_return'].values, 
        index=pd.to_datetime(result['month']),
        name="Benchmark (CORN)" # <-- Label atualizada
    ).dropna()

    # Gerar o relatório HTML (aceitando os gráficos de rolling zerados)
    qs.reports.html(
        strategy_returns_series, 
        benchmark=benchmark_series,
        output='nimbus_v2_backtest_report.html',
        title='Backtest Nimbus 2.2 (CORN)' # <-- Label atualizada
    )
    
    print(" Relatório de backtest salvo em nimbus_v2_backtest_report.html")

    # Mostrar métricas principais no console
    print("\nPrincipais Métricas do Backtest (Estratégia vs Benchmark):")
    print(f"Retorno Anualizado (Estratégia): {qs.stats.cagr(strategy_returns_series):.2%}")
    print(f"Retorno Anualizado (Benchmark): {qs.stats.cagr(benchmark_series):.2%}")
    print("---")
    print(f"Volatilidade Anualizada (Estratégia): {qs.stats.volatility(strategy_returns_series):.2%}")
    print(f"Volatilidade Anualizada (Benchmark): {qs.stats.volatility(benchmark_series):.2%}")
    print("---")
    print(f"Sharpe Ratio (Estratégia): {qs.stats.sharpe(strategy_returns_series):.2f}")
    print(f"Sharpe Ratio (Benchmark): {qs.stats.sharpe(benchmark_series):.2f}")
    print("---")
    print(f"Max Drawdown (Estratégia): {qs.stats.max_drawdown(strategy_returns_series):.2%}")
    print(f"Max Drawdown (Benchmark): {qs.stats.max_drawdown(benchmark_series):.2%}")