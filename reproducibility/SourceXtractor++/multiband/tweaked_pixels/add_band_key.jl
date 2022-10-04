l=filter( x-> startswith(x,"ds") & endswith(x, ".fits") & !occursin("psf",x), readdir() )
for f in l 
    band=replace(f, r".*ds_(\w+)\..*"=>s"\1" )
    run(`modhead $f BAND $band`)
end

