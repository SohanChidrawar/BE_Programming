import random

def checkCaptcha(captcha,user_captcha):
    if captcha == user_captcha:
        return True
    return False

def generateCaptcha(n):
    chrs = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOQRSTUVWXYZ0123456789"
    captcha = ""
    while(n):
        captcha += chrs[random.randint(1,1000) % 62]
        n -= 1
    return captcha

if __name__ == "__main__":
    # length of captcha
    n = 9
    captcha = generateCaptcha(n)
    print(captcha)

    user_captcha = input("Enter the captcha: ")
    if(checkCaptcha(captcha,user_captcha)):
        print("Captcha Matched!!")
    else:
        print("Captcha is Invalid...")

'''
# ----------------------------------------------------------------------------------------------
python .\captcha.py
2L07Bf5IF
Enter the captcha: 2L07Bf5IF
Captcha Matched!!

'''
