import numpy as np
import tushare as ts
from scipy.stats import t, norm
import matplotlib.pyplot as plt
import QuantLib as ql

def fat_tail_call_option_price(S0, K, T, r, nu, sigma, n_sims=100000):
    """肥尾分布下的欧式看涨期权定价（学生t分布假设）
    
    参数:
        S0 (float): 标的资产当前价格
        K (float): 行权价格
        T (float): 到期时间（年）
        r (float): 无风险利率
        nu (float): 学生t分布自由度（nu > 2）
        sigma (float): 年化波动率
        n_sims (int): 模拟路径数
    
    返回:
        float: 欧式看涨期权价格
    """
    if nu <= 2:
        raise ValueError("nu必须大于2以保证方差有限。")
    
    # 计算交易日总数和时间步长
    n_steps = int(round(T * 252))  # 总时间步数（按交易日计算）
    dt = 1.0 / 252                # 每日时间步长
    
    # 计算日波动率参数
    daily_var = sigma**2 / 252    # 日方差（无调整）
    risk_neutral_drift = (r - 0.5 * daily_var) * dt  # 风险中性漂移
    
    # 调整学生t分布的scale参数以匹配日方差
    adjusted_scale = np.sqrt(daily_var * (nu - 2) / nu)
    
    # 生成学生t分布收益
    t_returns = t.rvs(
        df=nu,
        loc=risk_neutral_drift,
        scale=adjusted_scale,
        size=(n_sims, n_steps)
    )
    
    # 计算累积收益并生成价格路径
    cumulative_returns = t_returns.cumsum(axis=1)
    paths = S0 * np.exp(cumulative_returns)
    
    # 计算到期payoff并贴现
    payoff = np.maximum(paths[:, -1] - K, 0)
    return np.exp(-r * T) * np.mean(payoff)

def fat_tail_put_option_price(S0, K, T, r, nu, sigma, n_sims=100000):
    """肥尾分布下的欧式看跌期权定价（学生t分布假设）
    
    参数:
        S0 (float): 标的资产当前价格
        K (float): 行权价格
        T (float): 到期时间（年）
        r (float): 无风险利率
        nu (float): 学生t分布自由度（nu > 2）
        sigma (float): 年化波动率
        n_sims (int): 模拟路径数
    
    返回:
        float: 欧式看跌期权的理论价格
    """
    if nu <= 2:
        raise ValueError("nu must be greater than 2 to have finite variance.")
    
    dt = T / 252  # 交易日日数假设
    # 调整scale以匹配波动率参数sigma
    adjusted_scale = sigma * np.sqrt(dt) * np.sqrt((nu - 2) / nu)
    # 生成学生t收益
    t_returns = t.rvs(nu, loc=(r - 0.5*sigma**2)*dt, scale=adjusted_scale, size=(n_sims, 252))
    # 计算价格路径
    paths = S0 * np.exp(t_returns.cumsum(axis=1))
    # 到期Payoff并贴现（看跌期权计算K - S_T）
    payoff = np.maximum(K - paths[:, -1], 0)
    return np.exp(-r * T) * np.mean(payoff)

def plot_fit(returns,etf_code,start_date,end_date):
    """Plot comparison of return distribution and fitted t-distribution"""
    plt.figure(figsize=(10, 6))
    
    # Plot histogram of returns
    plt.hist(returns, bins=50, density=True, alpha=0.6, 
             label='Actual Return Distribution')
    
    # Fit t-distribution to data
    nu, loc, scale = t.fit(returns)
    print(f"拟合得到的学生t分布自由度 nu = {nu:.2f}, 均值 loc = {loc:.4f}, 标准差 scale = {scale:.4f}")
    daily_var = scale**2 * (nu / (nu - 2))
    annual_sigma = np.sqrt(daily_var * 252)  # 年化波动率
    print(f"年化波动率 sigma = {annual_sigma:.4f}")
    
    # Plot fitted t-distribution
    x = np.linspace(returns.min(), returns.max(), 100)
    plt.plot(x, t.pdf(x, df=nu, loc=loc, scale=scale), 'r', lw=2, 
             label=f'Student t-distribution (ν={nu:.2f})')
    
    # Plot normal distribution for comparison
    plt.plot(x, norm.pdf(x, loc=np.mean(returns), scale=np.std(returns)), 'g--', lw=2, 
             label='Normal Distribution')
    
    plt.title(etf_code+' Return Distribution vs Fitted Models From '+start_date+" to "+end_date)
    plt.xlabel('Log Returns')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True)
    plt.savefig("./test_figure/"+etf_code+"_fat_tail_fit_"+start_date+"_to_"+end_date+".png")
    return nu

def quantlib_bs_option_price(S0, K, T, r, sigma, option_type=ql.Option.Call):
    """使用QuantLib计算BS模型下的期权价格
    
    参数:
        S0 (float): 标的资产当前价格
        K (float): 行权价格
        T (float): 到期时间（年）
        r (float): 无风险利率
        sigma (float): 年化波动率
        option_type: 期权类型(ql.Option.Call或ql.Option.Put)
    
    返回:
        float: 期权理论价格
    """
    # 设置计算日期
    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today
    
    # 创建期权参数
    option = ql.EuropeanOption(
        ql.PlainVanillaPayoff(option_type, K),
        ql.EuropeanExercise(today + ql.Period(int(T*365), ql.Days)))
    
    # 创建BS过程
    spot = ql.QuoteHandle(ql.SimpleQuote(S0))
    risk_free_curve = ql.YieldTermStructureHandle(
        ql.FlatForward(today, r, ql.Actual365Fixed()))
    dividend_curve = ql.YieldTermStructureHandle(
        ql.FlatForward(today, 0.0, ql.Actual365Fixed()))
    volatility = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(today, ql.NullCalendar(), sigma, ql.Actual365Fixed()))
    
    process = ql.BlackScholesMertonProcess(spot, dividend_curve, risk_free_curve, volatility)
    
    # 计算价格
    engine = ql.AnalyticEuropeanEngine(process)
    option.setPricingEngine(engine)
    
    return option.NPV()

if __name__ == "__main__":
    # 初始化Tushare
    pro = ts.pro_api("your_api_token") # 在这儿填你在https://tushare.pro/申请的api token

    ETF_list=['588000.SH','510050.SH','510300.SH','510500.SH','159915.SZ','512100.SH','510760.SH']

    for etf_code_single in ETF_list:
    # 用tushare接口查询etf_code在trading_dates中的价格
        etf_code = etf_code_single
        if etf_code_single != '512100.SH':
            fund_data = pro.fund_daily(ts_code=etf_code)
        else:
            fund_data = pro.fund_daily(ts_code=etf_code, start_date='20220905', end_date='20250521')
        # print(fund_data)
        # 取出第一个日期和最后一个日期
        end_date = fund_data['trade_date'].iloc[0]
        start_date = fund_data['trade_date'].iloc[-1]
        print(f"{etf_code} 的数据范围: {start_date} 到 {end_date}")

        # 从fund_data按日期顺序，从后往前，获取close列，存入数组中
        simulated_prices = np.array(fund_data['close'].tolist()[::-1])
        
        # 拟合学生t分布自由度，并绘制拟合效果图
        fitted_nu=plot_fit(np.diff(np.log(simulated_prices)),etf_code,start_date,end_date)

    # 使用QuantLib计算BS模型价格
    #bs_price = quantlib_bs_option_price(S0=1.045, K=1.300, T=35/365, r=0.016, sigma=0.2763+0.11)
    #print(f"QuantLib BS模型期权价格: {bs_price:.4f}")

    # 参数设置
    #price = fat_tail_call_option_price(S0=1.045, K=1.300, T=35/365, r=0.016, nu=100000, sigma=0.2763+0.11)
    #print(f"肥尾期权价格: {price:.4f}")