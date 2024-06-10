#4. Sukurkite klasę DateUtils su statiniais metodais, kad apskaičiuotumėte dienų
# skaičių tarp dviejų datų ir patikrintumėte, ar tam tikri metai yra keliamieji metai. Naudokite python paketą datetime.


from datetime import datetime

class DateUtils:
    @staticmethod
    def days_between_dates(date1, date2):
        delta = (date2 - date1)
        return delta

    @staticmethod
    def is_leap_year(year):
        return (year % 4 == 0)

# Example usage:
date1 = datetime(2024, 8, 1)
date2 = datetime(2024, 12, 31)

days_between = DateUtils.days_between_dates(date1, date2)
print(days_between)

year_to_check = 2024
if DateUtils.is_leap_year(year_to_check):
    print(year_to_check, "is a leap year.")
else:
    print(year_to_check, "is not a leap year.")