typhoid['DISTRICT'].fillna(typhoid['DISTRICT'].mode()[0], inplace = True)
typhoid['AGE'].fillna(typhoid['AGE'].median(), inplace = True)
typhoid['REPORT_VERIFIED'].fillna(typhoid['REPORT_VERIFIED'].mode()[0], inplace =
True)
typhoid['TEHSIL'].fillna(typhoid['TEHSIL'].mode()[0], inplace = True)
