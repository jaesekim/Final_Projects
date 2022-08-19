import streamlit as st

import pandas as pd
import numpy as np

from datetime import timedelta
from datetime import datetime

from bs4 import BeautifulSoup as bs
import requests

import FinanceDataReader as fdr
from pykrx import stock

import matplotlib.pyplot as plt
import koreanize_matplotlib


st.markdown("# Portfolio for Risk Averse")
st.markdown("## 무위험이자율")
st.markdown("* CD 91물 16년 1월 ~ 22년 연평균 수익률")
st.markdown("* 22년은 6월까지의 지표를 산술평균로 추정")
st.markdown("* 단위: %")
st.markdown("## 시장수익률")
st.markdown("* 2016년 ~ 2022년 (연간)")
st.markdown("* 22년은 모두 집계가 되지 않았기 때문에 7월 지표로 추정")
st.markdown("* 연평균 수익률 CAGR 사용")
st.markdown("## 주의사항")
st.markdown("* 위험 회피 성향을 지닌 투자자들에게 적합한 지표를 제공합니다.")
st.markdown("* 표기된 기대수익률은 연간 기대수익률 기준입니다.(단위: %)")
st.markdown("* 별표로 표시된 부분이 MVP, 최소분산포트폴리오지점입니다.")
st.markdown("* x표시가 된 부분이 모든 금액을 입력하신 종목에 투자했을 경우의 기대수익률입니다.")
st.markdown("* x표시를 기준으로 투자자의 성향에 따라 가중치를 조정해서 확인하시면 됩니다.")
st.markdown("* 해당 지표는 세금, 거래 수수료 등이 반영되지 않은 수치이므로 참고용으로 사용하시길 바랍니다.")

df_krx = fdr.StockListing("KRX")
df_krx = df_krx.dropna(axis=0).reset_index(drop=True)

tmp_item_info = st.text_input("Code Name", placeholder="종목명을 정확하게 입력해 주세요")

# 종목명입력하면 종목 코드와 시장 반환해 주는 함수
def find_history_krx(name):
    info_list = []
    
    
    code = df_krx.loc[df_krx["Name"] == name, "Symbol"].values[0]
    market = df_krx.loc[df_krx["Name"] == name, "Market"].values[0]
    info_list.append(name)
    info_list.append(code)
    info_list.append(market)
    
    return info_list

if tmp_item_info =='':
    st.text('검색하신 종목명이 없습니다.')
else:
    item_info = find_history_krx(tmp_item_info)

    # 무위험 이자율 
    # CD 91물 16년 1월 ~ 22년 연평균 수익률 (22년은 6월까지의 지표를 산술평균)
    # 단위 : %
    rf = 1.51

    # 시장수익률
    # 2016년 ~ 2022년 (연간) 22년은 모두 집계가 되지 않았기 때문에 7월로 대체
    # 연평균 수익률 CAGR 사용
    rm_kospi = 3.22
    rm_kosdaq = 4.10

    # 52주 베타 추출 함수
    def get_beta(code):
        response = requests.get(f"https://navercomp.wisereport.co.kr/v2/company/c1010001.aspx?cmp_cd={code}&cn=")
        html = bs(response.text, "lxml")
        tmp = html.select("#cTB11 > tbody > tr:nth-child(6) > td")
        
        
        return float(str(tmp[0]).split()[2])


    # 종목 기대수익률
    def expected_return(rm, rf, beta):
        return np.round(rf + beta * (rm - rf), 2)


    # 종목 표준편차
    def get_std(code):
        df = fdr.DataReader(code, "2016")["Close"]
        return np.std(df)


    # 종목 공분산
    def get_cov(code1, code2):
        df1 = fdr.DataReader(code1, "2016")["Close"]
        df2 = fdr.DataReader(code2, "2016")["Close"]
        if len(df1) != len(df2):
            if len(df1) > len(df2):
                df1 = fdr.DataReader(code1, df2.index[0])["Close"]
            else:
                df2 = fdr.DataReader(code1, df1.index[0])["Close"]
        return np.cov(df1, df2)[0][1]


    # 종목 상관계수
    def get_corr(code1, code2):
        cov = get_cov(code1, code2)
        return cov / (get_std(code1) * get_std(code2))

    # 고른 종목의 기대수익률과 표준편차
    input_item = []
    input_item.append(expected_return(rm_kospi, rf, get_beta(item_info[1])))
    input_item.append(get_std(item_info[1]))

    # fundamental 지표로 종목 정하기
    # PER, PBR, EPS, DIV, DPS, BPS
    # PER, PBR이 낮을 수록 저평가 돼어있다는 의미
    # 나머지 지표는 높을 수록 good
    df_per = stock.get_market_fundamental(datetime.today() - timedelta(1), market="ALL")
    df_per["Ticker"] = df_per.index
    df_per = df_per.reset_index(drop=True)

    # 0값 제외
    BPS = df_per['BPS'] > 0
    PER = df_per['PER'] > 0
    PBR = df_per['PBR'] > 0
    EPS = df_per['EPS'] > 0
    DIV = df_per['DIV'] > 0
    DPS = df_per['DPS'] > 0

    df_per = df_per[BPS & PER & PBR & EPS & DIV & DPS]

    # per 순위 매기기

    df_per = df_per.sort_values(by="PER", ascending=True).reset_index(drop=True)
    df_per["per_rank"] = df_per.index

    # pbr 순위 매기기

    df_per = df_per.sort_values(by="PBR", ascending=True).reset_index(drop=True)
    df_per["pbr_rank"] = df_per.index

    # eps 순위 매기기

    df_per = df_per.sort_values(by="EPS", ascending=False).reset_index(drop=True)
    df_per["eps_rank"] = df_per.index

    # DIV 순위 매기기

    df_per = df_per.sort_values(by="DIV", ascending=False).reset_index(drop=True)
    df_per["div_rank"] = df_per.index

    # DPS 순위 매기기

    df_per = df_per.sort_values(by="DPS", ascending=False).reset_index(drop=True)
    df_per["dps_rank"] = df_per.index

    # BPS 순위 매기기

    df_per = df_per.sort_values(by="BPS", ascending=False).reset_index(drop=True)
    df_per["bps_rank"] = df_per.index

    # 합산 점수가 가장 낮을 수록 높은 순위
    # 상위 50 종목
    df_per["total_rank"] = df_per["bps_rank"] + df_per["per_rank"] + df_per["pbr_rank"] + df_per["eps_rank"] + df_per["div_rank"] + df_per["dps_rank"]
    df_sorted = df_per.sort_values(by="total_rank", ascending=True).reset_index(drop=True).head(20)


    # 고른 종목과 상관계수 구하기
    corr = []
    for ticker in df_sorted["Ticker"]:
        corr.append(get_corr(item_info[1], ticker))
    df_sorted['corr'] = corr
    df_sorted = df_sorted.sort_values(by="corr", ascending=True).reset_index(drop=True)
    df_sorted["corr_rank"] = df_sorted.index

    # 상위 30개 종목 추리기

    df_sorted["total_rank"] = df_sorted["total_rank"] + df_sorted["corr_rank"]
    df_sorted = df_sorted.sort_values(by="total_rank", ascending=True).reset_index(drop=True).head(20)


    # 종목코드에 종목명 컬럼 매치
    Name = []
    for i in df_sorted["Ticker"]:
        Name.append(stock.get_market_ticker_name(i))
    df_sorted["Name"] = Name

    # 포트폴리오 계산 용 데이터프레임
    df_pf = df_sorted[['Ticker','Name', "corr", 'total_rank']]


    # 선정된 목록 기대수익률, 표준편차
    pf_return = []
    pf_std = []
    for j in df_pf["Ticker"]:
        pf_return.append(expected_return(rm_kospi, rf, get_beta(j)))
        pf_std.append(get_std(j))
    df_pf["E_return"] = pf_return
    df_pf["std"] = pf_std


    # 포트폴리오의 기대수익률과 표준편차 함수
    def portfolio_return(w1, r1, r2):
        return (w1 * r1) + ((1 - w1) * r2)

    def portfolio_std(w1, std1, std2, corr):
        return (w1 ** 2) * (std1 ** 2) + ((1 - w1) ** 2) * (std2 ** 2) + (2 * w1 * (1 - w1) * corr * std1 * std2)


    # MVP 조건 만족 하는 종목 추출
    # ρ < σ2 / σ1. 단, σ1 > σ2

    delete_item = []
    for i in range(len(df_pf)):
        stock_ticker = df_pf.iloc[i]["Ticker"]
        stock_std = df_pf.iloc[i]["std"]
        stock_corr = df_pf.iloc[i]["corr"]
        if input_item[1] > stock_std:
            if stock_corr >= (stock_std / input_item[1]):
                delete_item.append(stock_ticker)
        else:
            if stock_corr >= (input_item[1] / stock_std):
                delete_item.append(stock_ticker)

    # 조건에 만족하는 상위 다섯 개 종목 추출
    df_top5 = df_pf.loc[~df_pf["Ticker"].isin(delete_item)].reset_index(drop=True).head()


    # 최소분산포트폴리오 구성비 구하기
    # w2 = 1 - w1

    def get_mvp_weight(std1, std2, corr):
        numerator = (std2 ** 2) - (corr * std1 * std2)
        denominator = (std1 ** 2) + (std2 ** 2) - (2 * corr * std1 * std2)
        w1 = numerator / denominator
        
        return w1

    # w1, 즉 가중치의 첫번째 것이 입력받은 주식의 가중치이다.
    if len(df_top5) == 0:
        st.write("입력하신 종목과 연결할 적합한 종목을 찾을 수 없습니다.")
        st.write("다른 종목을 입력해 주십시오")

    for i in range(len(df_top5)):
        top5_name = df_top5.iloc[i]["Name"]
        top5_return = df_top5.iloc[i]["E_return"]
        top5_std = df_top5.iloc[i]["std"]
        top5_corr = df_top5.iloc[i]["corr"]
        tmp_return_2 = []
        tmp_std_2 = []
        for x in range(10000):
            weights_2 = np.random.random(2)
            weights_2 /= np.sum(weights_2)  # 가중치 합 1
            mvp_weight = get_mvp_weight(input_item[1], top5_std, top5_corr)
            
            p_return_2 = portfolio_return(weights_2[0], input_item[0], top5_return)
            p_std_2 = portfolio_std(weights_2[0], input_item[1], top5_std, top5_corr)
            
            mvp_return = portfolio_return(mvp_weight, input_item[0], top5_return)
            mvp_std = portfolio_std(mvp_weight, input_item[1], top5_std, top5_corr)
            
            tmp_return_2.append(p_return_2)
            tmp_std_2.append(p_std_2)
            
        fig = plt.figure(figsize=(20 , 10))
        plt.title(top5_name)
        plt.scatter(tmp_std_2, tmp_return_2)
        plt.scatter(mvp_std, mvp_return,c='r', marker='*', s=150)
        plt.scatter(portfolio_std(1, input_item[1], top5_std, top5_corr), 
                    portfolio_return(1, input_item[0], 
                    top5_return),c='c', marker='X', s=150)
        plt.xlabel("std(σ)")
        plt.ylabel("Expected rate of return(E[r])")
        st.pyplot(fig) 
        
        st.write(f"{item_info[0]} & {top5_name}로 구성된 MVP 연간 기대수익률: {np.round(mvp_return, 2)}%")
        st.write(f"{item_info[0]}의 연간 기대수익률: {input_item[0]}%")
        st.write(f"{item_info[0]}의 보유 비중: {np.round(mvp_weight, 2)}")
        st.write(f"{top5_name}의 보유 비중: {np.round(1 - mvp_weight, 2)}")