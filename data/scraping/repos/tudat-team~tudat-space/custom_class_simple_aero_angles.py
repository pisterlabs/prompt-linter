class LinearTimeAerodynamicGuidance:

    def __init__(self, angle_of_attack_rate, sideslip_angle_rate, bank_angle_rate, reference_time ):
	
	# Define members
        self.angle_of_attack_rate = angle_of_attack_rate
	self.sideslip_angle_rate = sideslip_angle_rate
	self.bank_angle_rate = bank_angle_rate
	self.reference_time = reference_time

    def get_current_aerodynamic_angles(self, current_time: float):

	if( current_time == current_time ):

            # Update the class to the current time
            angle_of_attack = self.angle_of_attack_rate * ( current_time - self.reference_time )
            sideslip_angle = self.sideslip_angle_rate * ( current_time - self.reference_time )
            bank_angle = self.bank_angle_rate * ( current_time - self.reference_time )


# Define angle function (required for input to rotation settings)   
guidance_class = LinearTimeAerodynamicGuidance( 
	angle_of_attack_rate = np.deg2rad( 0.01 ), 
	sideslip_angle_rate = 0.0,
	bank_angle_rate = np.deg2rad( 0.2 ),
	reference_time = 10.0 * tudatpy.constants.JULIAN_YEAR )

# Create settings for rotation model from guidance class
rotation_model_settings = environment_setup.rotation_model.aerodynamic_angle_based(
        central_body="Earth",
        target_frame = "VehicleFixed",
        angle_funcion = guidance_class.get_current_aerodynamic_angles ) 
