## simple julia script to be used to parse all fits files that are not PSFs
## and add BAND keyword according to filename

l=filter( x-> endswith(x, ".fits") & !occursin("psf",x), readdir() )
for f in l 
    band=replace(f, r".*ds_(\w+)\..*"=>s"\1" )
    run(`modhead $f BAND $band`)
end

