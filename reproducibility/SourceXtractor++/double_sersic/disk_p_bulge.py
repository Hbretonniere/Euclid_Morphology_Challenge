
# disk radius [pix] plus a prior on the size distribution
disk_radius = FreeParameter(lambda o: o.get_radius(), Range(lambda v, o: (.1 * v, 10. * v), RangeType.EXPONENTIAL))
X_disk_radius = DependentParameter(lambda r: np.log(r), disk_radius)
add_prior(X_disk_radius, 0.65, 0.45)
# disk ratio plus a prior on the distribution
disk_ratio = FreeParameter(lambda o: o.get_aspect_ratio(), Range((0.1, 1.0), RangeType.EXPONENTIAL))
X_disk_ratio = DependentParameter(lambda r: np.log((r-0.01)/(0.99-r)), disk_ratio)
add_prior(X_disk_ratio, 0.03, 1.0)

# bulge radius [pix] plus a prior on the size distribution
bulge_radius = FreeParameter(lambda o: o.get_radius(), Range(lambda v, o: (.05 * v, 5. * v), RangeType.EXPONENTIAL))
X_rel_size = DependentParameter( lambda x,y : np.log10(y)-(2*np.log10(x)-0.48) , disk_radius, bulge_radius) 
add_prior(X_rel_size, -0.9, 0.5 )
# bulge ratio plus a prior on the distribution
bulge_ratio = FreeParameter(0.7, Range((0.1, 1.0), RangeType.EXPONENTIAL))
X_bulge_ratio = DependentParameter(lambda r: np.log((r-0.01)/(0.99-r)), bulge_ratio)
add_prior(X_bulge_ratio, 0.5, 1.1)

angle = FreeParameter(lambda o: o.get_angle(), Range((-2 * 3.14159, 2 * 3.14159), RangeType.LINEAR))
angle_deg = DependentParameter(lambda x: np.fmod(x,np.pi/2.0)/np.pi*180.0, angle)

flux_total = get_flux_parameter()
x,y = get_pos_parameters()
ra,dec = get_world_position_parameters(x, y)

bulge_fract = FreeParameter(0.5, Range((1.0e-03,1.0), RangeType.LINEAR))
flux_bulge = DependentParameter(lambda f, r: f*r, flux_total, bulge_fract)
flux_disk = DependentParameter(lambda f, r: f*(1.0-r), flux_total, bulge_fract)
X_bulge_fract =  DependentParameter(lambda bf: np.log((bf+0.0001)/(1.0001-bf)), bulge_fract)
add_prior(X_bulge_fract, -2.3, 1.7)
mag = DependentParameter(lambda f: -2.5 * np.log10(f) + MAG_ZEROPOINT, flux_total)
add_model(mesgroup, ExponentialModel(x, y, flux_disk, disk_radius, disk_ratio, angle))
add_model(mesgroup, DeVaucouleursModel(x, y, flux_bulge, bulge_radius, bulge_ratio, angle))

add_output_column('x', x)
add_output_column('y', y)
add_output_column('mag', mag)
add_output_column('bt', bulge_fract)
add_output_column('disk_effR_px', disk_radius)
add_output_column('bulge_effR_px', bulge_radius)
add_output_column('angle', angle)
add_output_column('angle_deg', angle_deg)
add_output_column('disk_axr', disk_ratio)
add_output_column('bulge_axr', bulge_ratio)

add_output_column('RA', ra)
add_output_column('Dec', dec)
add_output_column('rel_s', X_rel_size)
add_output_column('X_bt', X_bulge_fract)
add_output_column('X_rel_size', X_rel_size)
add_output_column('X_disk_radius', X_disk_radius)
add_output_column('X_disk_axr', X_disk_ratio)
add_output_column('X_bulge_axr', X_bulge_ratio)
add_output_column('flux_disk', flux_disk)
add_output_column('flux_bulge', flux_bulge)
add_output_column('flux_tot', flux_total)
