def time_to_hms(time):
	hours, rem = divmod(time, 3600)
	minutes, seconds = divmod(rem, 60)
	return hours, minutes, seconds

def time_to_hms_string(time):
	hours, minutes, seconds = time_to_hms(time)
	return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)