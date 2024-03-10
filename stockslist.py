# -*- coding: utf-8 -*-
class Stocklist:

    def __init__(self):
        self.amostra_acoes = [
    "ABEV3",
    "BEEF3",
    "JBSS3",
    "ITSA4",
    "PETR3"

]
        self.list_completa_acoes = [
    "DXCO3",
    "PETZ3",
    "RRRP3",
    "RECV3",
    "PRIO3",
    "SMTO3",
    "CSNA3",
    "CVCB3",
    "USIM5",
    "TIMS3",
    "CYRE3",
    "RDOR3",
    "ENEV3",
    "BRFS3",
    "TRPL4",
    "HAPV3",
    "NTCO3",
    "CSAN3",
    "AZUL4",
    "EMBR3",
    "ABEV3",
    "EZTC3",
    "CMIG4",
    "MRVE3",
    "B3SA3",
    "BEEF3",
    "ALPA4",
    "JBSS3",
    "CPLE6",
    "ITSA4",
    "SLCE3",
    "ITUB4",
    "UGPA3",
    "GGBR4",
    "SBSP3",
    "BBDC4",
    "SUZB3",
    "MGLU3",
    "HYPE3",
    "MULT3",
    "CRFB3",
    "RAIL3",
    "CIEL3",
    "ELET3",
    "BBDC3",
    "SOMA3",
    "ARZZ3",
    "TOTS3",
    "ELET6",
    "BHIA3",
    "WEGE3",
    "GOAU4",
    "ALOS3",
    "RAIZ4",
    "RENT3",
    "LREN3",
    "RADL3",
    "LWSA3",
    "CCRO3",
    "VBBR3",
    "EQTL3",
    "ASAI3",
    "EGIE3",
    "BBSE3",
    "VIVT3",
    "PCAR3",
    "CPFE3",
    "MRFG3",
    "VALE3",
    "BRAP4",
    "BBAS3",
    "IRBR3",
    "BRKM5",
    "YDUQ3",
    "CMIN3",
    "COGN3",
    "VAMO3",
    "FLRY3",
    "PETR3",
    "PETR4"
]
        self.cripto = [
            'BTC-USD', 
                       'ETH-USD', 'USDT-USD', 'BNB-USD', 'USDC-USD', 'XRP-USD', 'SOL1-USD', 'DOT1-USD', 'DOGE-USD', 'AVAX-USD']
    
    
    def get_list_stocks(self):
        inicializado = False
        while not inicializado:
            try:
                analise =  int(input("Qual Analise prefere: 1 =  Amostra de Ações | 2 = Lista Completa Ações Ibovespa | 3 = Criptomoedas (10 principais criptos): "))
                if analise == 1:
                    acoes_SA = [acao + ".SA" for acao in self.amostra_acoes]
                    inicializado = True
                    return acoes_SA
                elif analise == 2:
                    acoes_SA = [acao + ".SA" for acao in self.list_completa_acoes]
                    inicializado = True
                    return acoes_SA
                elif analise == 3:
                    acoes_SA = [acao + ".SA" for acao in self.cripto]
                    inicializado = True
                    return acoes_SA
                else:
                    print("Digite somente 1 ou 2")
            except ValueError:
                print("Por favo digite somente número!")


    

