#**********************************************************
#Title      :SimplexMethod
#Author     :Tanabe Yuta
#Date       :2019/10/16
#Copyright  :(C)2019 TanabeYuta
#**********************************************************


import numpy as np


#**********************************************************
#最適化問題の定義
#       Maximize    :Fx
#       Subject to  :Gx=0   (k本)
#                    Hpx>=h (l本)
#                    Hmx<=h (m本)
#                    xj>=0  (n本)
#**********************************************************


#**********シンプレックス法のフロー部分**********
def SimplexMethod(_F, _G = None, _g = None, _H = None, _h = None):
    if(_G is None):
        _G = np.empty((0, _F.shape[0]))
        _g = np.empty(0)
    elif(_H is None):
        _H = np.empty((0, _F.shape[0]))
        _h = np.empty(0)

    hpindex = np.where(_h <= 0)
    hmindex = np.where(_h > 0)

    Hp = -_H[hpindex]
    hp = -_h[hpindex]
    Hm = _H[hmindex]
    hm = _h[hmindex]

    n = _F.shape[0]     #設計変数ベクトルの次元
    k = _G.shape[0]     #等式制約ベクトルの次元
    l = Hp.shape[0]     #逆向き不等式制約ベクトルの次元
    m = Hm.shape[0]     #順向き不等式制約ベクトルの次元

    #----------第一段階----------
    A1 = np.block([[Hm, np.eye(m), np.zeros((m, l)), np.zeros((m, l)), np.zeros((m, k))], [Hp, np.zeros((l, m)), -np.eye(l), np.eye(l), np.zeros((l, k))], [_G, np.zeros((k, m)), np.zeros((k, l)), np.zeros((k, l)), np.eye(k)]])
    b1 = np.block([hm, hp, _g])
    c1 = np.block([np.zeros(n+m+l), -np.ones(l+k)])
    xbindex = np.block([np.arange(n, n+m, 1, dtype = 'int'), np.arange(n+m+l, n+m+2*l+k, 1, dtype = 'int')]) 
    Binv = np.eye(m+l+k)
    status, Binv, b2 = SimplexCore(A1, b1, c1, Binv, xbindex)
    
    #----------第二段階----------
    #←要修正:問題が不能か判定する
    A2 = A1[:,:n+m+l]
    c2 = np.block([_F, np.zeros(m+l)])
    status, Binv, beta = SimplexCore(A2, b2, c2, Binv, xbindex)
    xopt = np.zeros(n+m+l)
    xopt[xbindex] = beta
    return xopt[:n]


#**********シンプレックス法のコア部分**********
def SimplexCore(_A, _beta, _c, _Binv, _xbindex):
    m = _A.shape[0]           
    xnbindex = np.setdiff1d(np.arange(0, _A.shape[1], 1, dtype = 'int'), _xbindex)
    status = 1
        
    while(1):
        #リデューストコストの計算
        cbar = np.dot(np.dot(_c[_xbindex], _Binv), _A[:,xnbindex]) - _c[xnbindex]
        
        #最適解か判定
        if(np.amin(cbar) >= 0.0):
            status = 0
            break

        #変更する非基底変数を選択
        t = cbar.argmin()
        s = xnbindex[t]
        
        #基底表示を求める
        alphas = np.dot(_Binv, _A[:, s])

        #非有界か判定
        if(np.amax(alphas) < 0.0):
            status = -1
            break

        #変更する基底変数を選択←要修正
        alphasmod = np.copy(alphas)
        alphasmod[alphasmod < 0.0] = 0.0
        r = (_beta / alphasmod).argmin()

        #基底解を更新
        tmp = _xbindex[r]
        _xbindex[r] = s
        xnbindex[t] = tmp

        #基底逆行列を更新
        E = np.eye(m)
        alphars = alphas[r]
        E[:,r] = -alphas/alphars
        E[r,r] = 1.0/alphars 
        _Binv = np.dot(E, _Binv)

        #基底解の更新
        _beta = np.dot(E, _beta)

    #結果を返す
    return status, _Binv, _beta


#**********サンプルの実行**********
if __name__ == "__main__":
    #----------制約条件の定義----------
    G = np.zeros((1, 2));   g = np.zeros(1)
    G[0][0] = 1.0;  G[0][1] = -1.0; g[0] = 15.0

    H = np.zeros((2, 2));  h = np.zeros(2)
    H[0][0] = -2.0; H[0][1] = -5.0; h[0] = -30.0
    H[1][0] = 2.0;  H[1][1] = 1.0;  h[1] = 60.0
    
    #----------目的関数の定義----------
    F = np.zeros(2)
    F[0] = 6.0; F[1] = 7.0

    #----------シンプレックス法の実行----------
    x = SimplexMethod(F, G, g, H, h)
    print(x)