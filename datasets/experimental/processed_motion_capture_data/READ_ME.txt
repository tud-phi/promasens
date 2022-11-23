-Original convention for the database was:

	Format Version	1.23	Take Name	Take 2021-12-07 01.15.11 PM	Take Notes		Capture Frame Rate	50	Export Frame Rate	50	Capture Start Time	2021-12-07 01.15.11.095 PM	Capture Start Frame	255570	Total Frames in Take	3100	Total Exported Frames	3100	Rotation Type	Quaternion	Length Units	Meters	Coordinate Space	Global
																								
		Type	Rigid Body	Rigid Body	Rigid Body	Rigid Body	Rigid Body	Rigid Body	Rigid Body	Rigid Body	Rigid Body	Rigid Body	Rigid Body	Rigid Body	Rigid Body	Rigid Body	Rigid Body	Rigid Body						
		Name	Robot base	Robot base	Robot base	Robot base	Robot base	Robot base	Robot base	Robot base	Ring	Ring	Ring	Ring	Ring	Ring	Ring	Ring						
		ID	F5D2F39756B011ECD639B19196B4A0B1	F5D2F39756B011ECD639B19196B4A0B1	F5D2F39756B011ECD639B19196B4A0B1	F5D2F39756B011ECD639B19196B4A0B1	F5D2F39756B011ECD639B19196B4A0B1	F5D2F39756B011ECD639B19196B4A0B1	F5D2F39756B011ECD639B19196B4A0B1	F5D2F39756B011ECD639B19196B4A0B1	72A770A2575211ECD639B19196B4A1B1	72A770A2575211ECD639B19196B4A1B1	72A770A2575211ECD639B19196B4A1B1	72A770A2575211ECD639B19196B4A1B1	72A770A2575211ECD639B19196B4A1B1	72A770A2575211ECD639B19196B4A1B1	72A770A2575211ECD639B19196B4A1B1	72A770A2575211ECD639B19196B4A1B1						
		Rotation	Rotation	Rotation	Rotation	Position	Position	Position	Mean Marker Error	Rotation	Rotation	Rotation	Rotation	Position	Position	Position	Mean Marker Error						
	Frame	Time (Seconds)	X	Y	Z	W	X	Y	Z		X	Y	Z	W	X	Y	Z							
			

-Is then later changed for processing into:
				X_b	Y_b	Z_b	W_b	TX_b	TY_b	TZ_b	mean	X_r	Y_r	Z_r	W_r	TX_r	TY_r	TZ_r	mean
