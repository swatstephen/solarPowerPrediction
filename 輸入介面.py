#[5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5,15.5,17]Columns: [a,b,c,d,e,f,g,h,i,j,k]
#[氣象_多雲, 氣象_多雲有雷, 氣象_晴, 氣象_陰, 氣象_陰有雨]  

#實際應用  #10/19 外氣25.8 陰 [0,0,1,0] 下午一點[0,0,0,1,0,0,0,0,0,0]
#x_real = np.array([[30.8,0,1,0,0,0,0,0,0,0,0,0,1,0,0]])
def enc(name,k,tmp):
    import bisect

    ls = [6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5,15.5,16.5]
    n = []
    k = k
    w = ['多雲', '多雲有雷', '晴', '有雨','陰', '陰有雨']
    t = []
    position = bisect.bisect_left(ls[::1], k)
    rank = len(ls)-position
    #print(rank)

    name = name
    for j in w:
        if j == name:
            t.append(1)
        else:
            t.append(0)
    t.pop()        
    #print(t)      

    for i in range(1,len(ls)):
        if i == rank-1 :
            n.append(1)
        else:
            n.append(0)
    n.pop()
    n.reverse()
    #print(n)    

    ata = t + n 
    #print(ata)
    
    tmp = tmp

    ata.insert(0,tmp)
    

    return ata

#print(enc('晴',11,31.5))
