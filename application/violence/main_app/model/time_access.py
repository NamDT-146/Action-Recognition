class TimeAccess(object):
    def __init__(self):
        self.sundayStart1 = "00:00:00"
        self.sundayEnd1 = "23:59:59"
        self.sundayStart2 = "00:00:00"
        self.sundayEnd2 = "23:59:59"
        self.sundayStart3 = "00:00:00"
        self.sundayEnd3 = "23:59:59"
        self.sundayStart4 = "00:00:00"
        self.sundayEnd4 = "23:59:59"

        self.mondayStart1 = "00:00:00"
        self.mondayEnd1 = "23:59:59"
        self.mondayStart2 = "00:00:00"
        self.mondayEnd2 = "23:59:59"
        self.mondayStart3 = "00:00:00"
        self.mondayEnd3 = "23:59:59"
        self.mondayStart4 = "00:00:00"
        self.mondayEnd4 = "23:59:59"

        self.tuesdayStart1 = "00:00:00"
        self.tuesdayEnd1 = "23:59:59"
        self.tuesdayStart2 = "00:00:00"
        self.tuesdayEnd2 = "23:59:59"
        self.tuesdayStart3 = "00:00:00"
        self.tuesdayEnd3 = "23:59:59"
        self.tuesdayStart4 = "00:00:00"
        self.tuesdayEnd4 = "23:59:59"

        self.wednesdayStart1 = "00:00:00"
        self.wednesdayEnd1 = "23:59:59"
        self.wednesdayStart2 = "00:00:00"
        self.wednesdayEnd2 = "23:59:59"
        self.wednesdayStart3 = "00:00:00"
        self.wednesdayEnd3 = "23:59:59"
        self.wednesdayStart4 = "00:00:00"
        self.wednesdayEnd4 = "23:59:59"

        self.thursdayStart1 = "00:00:00"
        self.thursdayEnd1 = "23:59:59"
        self.thursdayStart2 = "00:00:00"
        self.thursdayEnd2 = "23:59:59"
        self.thursdayStart3 = "00:00:00"
        self.thursdayEnd3 = "23:59:59"
        self.thursdayStart4 = "00:00:00"
        self.thursdayEnd4 = "23:59:59"

        self.fridayStart1 = "00:00:00"
        self.fridayEnd1 = "23:59:59"
        self.fridayStart2 = "00:00:00"
        self.fridayEnd2 = "23:59:59"
        self.fridayStart3 = "00:00:00"
        self.fridayEnd3 = "23:59:59"
        self.fridayStart4 = "00:00:00"
        self.fridayEnd4 = "23:59:59"

        self.saturdayStart1 = "00:00:00"
        self.saturdayEnd1 = "23:59:59"
        self.saturdayStart2 = "00:00:00"
        self.saturdayEnd2 = "23:59:59"
        self.saturdayStart3 = "00:00:00"
        self.saturdayEnd3 = "23:59:59"
        self.saturdayStart4 = "00:00:00"
        self.saturdayEnd4 = "23:59:59"

    @staticmethod
    def deserialize(data) -> "TimeAccess":
        # print('------------------------',data)
        new_time_acess = TimeAccess()
        try:
            new_time_acess.sundayStart1 = data["sundayStart1"]
            new_time_acess.sundayEnd1 = data["sundayEnd1"]
            new_time_acess.sundayStart2 = data["sundayStart2"]
            new_time_acess.sundayEnd2 = data["sundayEnd2"]
            new_time_acess.sundayStart3 = data["sundayStart3"]
            new_time_acess.sundayEnd3 = data["sundayEnd3"]
            new_time_acess.sundayStart4 = data["sundayStart4"]
            new_time_acess.sundayEnd4 = data["sundayEnd4"]

            new_time_acess.mondayStart1 = data["mondayStart1"]
            new_time_acess.mondayEnd1 = data["mondayEnd1"]
            new_time_acess.mondayStart2 = data["mondayStart2"]
            new_time_acess.mondayEnd2 = data["mondayEnd2"]
            new_time_acess.mondayStart3 = data["mondayStart3"]
            new_time_acess.mondayEnd3 = data["mondayEnd3"]
            new_time_acess.mondayStart4 = data["mondayStart4"]
            new_time_acess.mondayEnd4 = data["mondayEnd4"]

            new_time_acess.tuesdayStart1 = data["tuesdayStart1"]
            new_time_acess.tuesdayEnd1 = data["tuesdayEnd1"]
            new_time_acess.tuesdayStart2 = data["tuesdayStart2"]
            new_time_acess.tuesdayEnd2 = data["tuesdayEnd2"]
            new_time_acess.tuesdayStart3 = data["tuesdayStart3"]
            new_time_acess.tuesdayEnd3 = data["tuesdayEnd3"]
            new_time_acess.tuesdayStart4 = data["tuesdayStart4"]
            new_time_acess.tuesdayEnd4 = data["tuesdayEnd4"]

            new_time_acess.wednesdayStart1 = data["wednesdayStart1"]
            new_time_acess.wednesdayEnd1 = data["wednesdayEnd1"]
            new_time_acess.wednesdayStart2 = data["wednesdayStart2"]
            new_time_acess.wednesdayEnd2 = data["wednesdayEnd2"]
            new_time_acess.wednesdayStart3 = data["wednesdayStart3"]
            new_time_acess.wednesdayEnd3 = data["wednesdayEnd3"]
            new_time_acess.wednesdayStart4 = data["wednesdayStart4"]
            new_time_acess.wednesdayEnd4 = data["wednesdayEnd4"]

            new_time_acess.thursdayStart1 = data["thursdayStart1"]
            new_time_acess.thursdayEnd1 = data["thursdayEnd1"]
            new_time_acess.thursdayStart2 = data["thursdayStart2"]
            new_time_acess.thursdayEnd2 = data["thursdayEnd2"]
            new_time_acess.thursdayStart3 = data["thursdayStart3"]
            new_time_acess.thursdayEnd3 = data["thursdayEnd3"]
            new_time_acess.thursdayStart4 = data["thursdayStart4"]
            new_time_acess.thursdayEnd4 = data["thursdayEnd4"]

            new_time_acess.fridayStart1 = data["fridayStart1"]
            new_time_acess.fridayEnd1 = data["fridayEnd1"]
            new_time_acess.fridayStart2 = data["fridayStart2"]
            new_time_acess.fridayEnd2 = data["fridayEnd2"]
            new_time_acess.fridayStart3 = data["fridayStart3"]
            new_time_acess.fridayEnd3 = data["fridayEnd3"]
            new_time_acess.fridayStart4 = data["fridayStart4"]
            new_time_acess.fridayEnd4 = data["fridayEnd4"]

            new_time_acess.saturdayStart1 = data["saturdayStart1"]
            new_time_acess.saturdayEnd1 = data["saturdayEnd1"]
            new_time_acess.saturdayStart2 = data["saturdayStart2"]
            new_time_acess.saturdayEnd2 = data["saturdayEnd2"]
            new_time_acess.saturdayStart3 = data["saturdayStart3"]
            new_time_acess.saturdayEnd3 = data["saturdayEnd3"]
            new_time_acess.saturdayStart4 = data["saturdayStart4"]
            new_time_acess.saturdayEnd4 = data["saturdayEnd4"]

        except Exception as e:
            print('------------------------',e)

        return new_time_acess
