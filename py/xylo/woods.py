from xylo.types import Wood

tasmanian_oak = Wood.make_E_G(E = 6.067e9, G = 16.0303e9, rho = 611.0152)

# https://vicbeam.com.au/product-information/
# E = 21,000MPa; G = 1,400MPa, rho = 990kg/m^3
# https://www.wood-database.com/spotted-gum/
# E = 19.77GPa; rho = 940kg/m^3
# spotted_gum = Wood.make_E_G(E = 21.000e9, G = 1.4e9, rho = 990)
# A Comparative Study of using Static and Ultrasonic Material Testing Methods to Determine the Anisotropic Material Properties of Wood, Dackerman et al
spotted_gum = Wood.make_E_nu(E = 26.100e9, nu = 0.49, rho = 990)

# eastern indian rosewood
# https://www.matweb.com/search/datasheet.aspx?matguid=e3b5a2a1a6794cddb47a91fbea57c18a&ckck=1
# eastern_indian_rosewood = Wood(E = 

# Eucalyptus hemilampra
# https://www.matweb.com/search/DataSheet.aspx?MatGUID=1022bc4dd73149399e4ad53d386cf458
# Eucalyptus globulus
# https://www.matweb.com/search/DataSheet.aspx?MatGUID=a246917e73594287b60e9338f85d511e

# From supermediocre http://supermediocre.org/index.php/rich/richs-projects/xylophone-project/
# rosewood = Wood.make_E_G(E = 23.7e9, G = 19e9, rho = 1082)

# From Beaton and Scavone (2021), x-plane, xz plane
# E  = [ 23, 2.3, 1.15 ] (GPa)
# nu = xy: 0.30, yz: 0.60, xz: 0.45
# G  = xy: 3.0,  yz: 1.0,  xz: 3.0 (GPa)
rosewood = Wood.make_E_G(E = 23e9, G = 3.0e9, rho = 1080)


# From Beaton and Scavone 
aluminium = Wood.make_E_G(E = 68.9e9, G = 25.9e9, rho = 2700)

# https://apacinfrastructure.com.au/material-specifications-6060-t5-aluminium-alloy
aluminium_6060T5 = Wood.make_E_G(E = 68.9e9, G = 25.8e9, rho = 2700)


# Test from Numerical simulations of xylophones. I. Time-domain modeling of the vibrating bars (Chaigne, Doutaut, 1997)
# Inferred properties from supermediocre
wood_test_chaigne1997 = Wood.make_E_nu(E = 9.54E9, nu = 0.3, rho = 1015)

mingming = Wood.make_E_G(E = 17.96e9, G = 6.75e9, rho = 837.187565)

# from Mingming: "Suits (2001) set the values of E=14.7GPa and G=E/10 to achieve acceptable results for oak bars"
suits_oak = Wood.make_E_G(E = 14.7e9, G = 1.47e9, rho = 760)
