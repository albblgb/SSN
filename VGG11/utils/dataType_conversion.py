import torch
import numpy as np

def list_to_tensor(list):
    return torch.from_numpy(np.array(list))
    

''' float32 to IEEE 754 binary string '''
def ConvertFixedIntegerToComplement(fixedInterger) :#浮点数整数部分转换成补码(整数全部为正)
    return bin(fixedInterger)[2:]
 
def ConvertFixedDecimalToComplement(fixedDecimal) :#浮点数小数部分转换成补码
    fixedpoint = int(fixedDecimal) / (10.0**len(fixedDecimal))
    s = ''
    while fixedDecimal != 1.0 and len(s) < 23 :
        fixedpoint = fixedpoint * 2.0
        s += str(fixedpoint)[0]
        fixedpoint = fixedpoint if str(fixedpoint)[0] == '0' else fixedpoint - 1.0
    return s
 
def ConvertToExponentMarker(number) : #阶码生成
    return bin(number + 127)[2:].zfill(8)
 
 
def ConvertFloatToBinary(floatingPoint) :#转换成IEEE754标准的数
    floatingPointString = str(floatingPoint)
    if floatingPointString.find('-') != -1 :#判断符号位
        sign = '1'
        floatingPointString = floatingPointString[1:]
    else :
        sign = '0'
    l = floatingPointString.split('.')#将整数和小数分离
    front = ConvertFixedIntegerToComplement(int(l[0]))#返回整数补码
    rear  = ConvertFixedDecimalToComplement(l[1])#返回小数补码
    floatingPointString = front + '.' + rear #整合
    relativePos =   floatingPointString.find('.') - floatingPointString.find('1')#获得字符1的开始位置
    if relativePos > 0 :#若小数点在第一个1之后
        exponet = ConvertToExponentMarker(relativePos-1)#获得阶码
        mantissa =  floatingPointString[floatingPointString.find('1')+1 : floatingPointString.find('.')]  + floatingPointString[floatingPointString.find('.') + 1 :] # 获得尾数
    else :
        exponet = ConvertToExponentMarker(relativePos)#获得阶码
        mantissa = floatingPointString[floatingPointString.find('1') + 1: ]  # 获得尾数
    mantissa =  mantissa[:23] + '0' * (23 - len(mantissa))
    floatingPointString = '0b' + sign + exponet + mantissa
    return floatingPointString
    # return hex( int( floatingPointString , 2 ) )
 
''' IEEE 754 binary string to float32 '''
def ConvertExponent(strData):#阶码转整数
    return int(strData,2)-127

def ConverComplementToFixedDecimal(fixedStr):#字符串转小数
    count=1
    num=0
    for ch in fixedStr:
        if ch=="1":
            num+=2**(-count)
        count+=1
    return num

def ConverComplementToInteger(fixedStr):#字符串转整数
    return int(fixedStr,2)
    
def ConvertBinartToFloat(binStr): #IEEE754 浮点字符串转float
	# if strData=="00000000":
    #     return 0.0
    # binStr="".join(hex2bin_map[i] for i in strData)
    sign = binStr[0]
    exponet=binStr[1:9]#阶码
    mantissa="1"+binStr[9:]#尾数
    fixedPos=ConvertExponent(exponet)
    if fixedPos>=0: #小数点在1后面
        fixedDec=ConverComplementToFixedDecimal(mantissa[fixedPos+1:])#小数转换
        fixedInt=ConverComplementToInteger(mantissa[:fixedPos+1])#整数转换
    else: #小数点在1前面（原数在[-0.99,0.99]范围内)
        mantissa="".zfill(-fixedPos)+mantissa
        fixedDec=ConverComplementToFixedDecimal(mantissa[1:])#小数转换
        fixedInt=ConverComplementToInteger(mantissa[0])#整数转换
    fixed=fixedInt+fixedDec
    if sign=="1":
        fixed=-fixed
    return fixed

# # a = -0.23
# b = 0.209687
# print(ConvertFloatToBinary(0.006689))
# # print(ConvertBinartToFloat(ConvertFloatToBinary(a)[2:])*2)
# print(ConvertBinartToFloat('01111011110110110010111100000000'))
