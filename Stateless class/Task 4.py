#4. Sukurkite klasę DateUtils su statiniais metodais, kad apskaičiuotumėte dienų
# skaičių tarp dviejų datų ir patikrintumėte, ar tam tikri metai yra keliamieji metai. Naudokite python paketą datetime.

import datetime
from datetime import datetime

class DateUtils:
    @staticmethod
    def days_between_dates(date1, date2):
        delta = abs(date2 - date1).days
        return delta

    @staticmethod
    def is_leap_year(year):
        return (year % 4 == 0)

# Example usage:
date1 = datetime(2024, 1, 1)
date2 = datetime(2024, 12, 31)

days_between = DateUtils.days_between_dates(date1, date2)
print("Days between", date1, "and", date2, ":", days_between)

year_to_check = 2024
if DateUtils.is_leap_year(year_to_check):
    print(year_to_check, "is a leap year.")
else:
    print(year_to_check, "is not a leap year.")