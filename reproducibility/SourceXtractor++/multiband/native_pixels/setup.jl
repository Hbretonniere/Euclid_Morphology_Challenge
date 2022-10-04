# using Pkg ; Pkg.add.( ["FITSIO", "CFITSIO", "Statistics", "StatsBase"] )
using DelimitedFiles, FITSIO, CFITSIO, Statistics, StatsBase
dir = "/raid/data/euclid_morpho_challenge/FIELD0/Double Sersic"
cases = Dict( "ssersic" => joinpath( dir, "Single Sersic") ,
              "dsersic" => joinpath( dir, "Double Sersic") ,
              "realm" => joinpath( dir, "Realistic Morphologies") )
bands = ( "y", "j", "h" )

########################################
function fsymlink(x,y)
    islink(y) && rm(y)
    symlink(x,y)
end
#######################################
function tweak_rms_ima( imaf, rmsf, newimaf, newrmsf ; flatten=false, rms_margin::Real=0.01, sigma_ratio::Real=10.0f0 )
    io = FITS(imaf,"r")
    imav = read(io[1])
    imah = read_header(io[1])
    close(io)
    io=FITS(rmsf,"r")
    rmsv= read(io[1])
    rmsh = read_header(io[1])

    low_t = 0.1*median(rmsv)   # search min rms where significant
    min_rms = minimum(rmsv[ findall( rmsv .> low_t ) ] )
    thresh=min_rms*(1+rms_margin)   # identify sky areas in rms map
    blank_pixels = findall( low_t .< rmsv .< thresh )
    μ,σ = mean_and_std( imav[ blank_pixels ] )  # basic image statistics in that area

    #rmsv .*= (σ/min_rms)  ### rescale image correctly!
    @show imaf, σ, min_rms

    ### now fit for bg and gain!!! sig^2 = v[1] + v[2] * ima = (bg+ima)/g
    thresh = σ*3000
    ok_pixels = findall( low_t .< rmsv .< thresh )
    nv=length(ok_pixels)
    M_mat = [ ones(nv) max.(imav[ok_pixels],0) ]
    @time v = ( M_mat' * M_mat ) \ ( M_mat' * ( rmsv[ok_pixels].^2 ) )

    hv = sqrt( v[1] ) / sigma_ratio
    correct = 1 - 1/sigma_ratio^2

    imav .= ( rmsv.^2 .- v[1] ) ./ v[2] .+ randn(size(rmsv))*hv

    rmsv .= max.( sqrt.( rmsv.^2 .- v[1] .* correct ), 0.f0 )

    FITS(newimaf,"w") do io
        write(io, imav, header=rmsh )
        CFITSIO.fits_write_history( io.fitsfile, "New image without bg noise" )
    end
    FITS(newrmsf,"w") do io
        write(io, rmsv, header=rmsh )
        CFITSIO.fits_write_history( io.fitsfile, "New rms only made of source noise" )
    end
    v[1]/v[2], 1/v[2]
end
########################################
function read_ground_truth( ; gals=false )
    fn = gals ? joinpath(dir,cases["dsersic"],"vis_0_gals_subset.list") :
        joinpath(dir,cases["dsersic"],"vis_0_stars_subset.list")
    v,h = readdlm( fn, header=true )
    h = reshape( string.(h), length(h))
    popfirst!(h)
    @info h
    res=Dict{String,Any}()
    @info res
    for (i,k) in enumerate(h)
        res[k] = Float64.( v[:,i] )
        if k in ( "X", "Y" )
            res[k*"_nat"] = res[k] / 3
        end
    end
    res
end

#=
## To load ground truth catalogs
gt_s = read_ground_truth()
gt_g = read_ground_truth(gals=true)
FITS("jjj.fits","w") do io ; write(io,gt_s) ; end
FITS("ggg.fits","w") do io ; write(io,gt_g) ; end
=#

def_swarp="swarp -MEM_MAX 10000 -SUBTRACT_BACK N -OVERSAMPLING 1 -RESAMPLING_TYPE NEAREST -WEIGHTOUT_NAME /dev/null -WRITE_XML N "

bands = replace.( filter( x->occursin(r"dser.*\.fits$",x) & !occursin("rms.fits",x), readdir(dir)),
                  r"dsersic_0_(.*)\.fits" => s"\1" )

bg_v = Float64[]
gain_v = Float64[]

for b in bands
    ima=joinpath(dir,"dsersic_0_$b.fits") ; lima="ds_$b.fits"
    lima0=lima 
    @info b,ima
    wima=joinpath(dir,"dsersic_0_$b.rms.fits") ; lwima="ds_$b.rms.fits"
    lwima0=lwima 
    fsymlink(ima, lima)
    fsymlink(wima, lwima )
    if occursin("lsst",b) | occursin("nir",b)
        lima=replace(lima0,r".fits"=>".native.fits")
        sw_cmd="$def_swarp -HEADEROUT_NAME coadd_os"* (occursin("lsst",b) ? "2" : "3" ) *".head"
        run(`$(split(sw_cmd)) $lima0 -IMAGEOUT_NAME $lima`)
        #@info "$(split(sw_cmd)) $lima0 -IMAGEOUT_NAME $lima"
        lwima=replace(lwima0,r".fits"=>".native.fits")        
        run(`$(split(sw_cmd)) $lwima0 -IMAGEOUT_NAME $lwima`)
   end
    
    if b=="vis"
        fsymlink("psf_vis_os045_high_nu.psf", "psf_$b.fits")
        ## PSFEx format tells SE++ about increased sampling of this PSF model
    elseif occursin("lsst",b)
        sb="lsst_"*b[1]
        cp( joinpath(dir,"../../PSFs/psf_$(sb)12jun.swarp.resamp_centered.norm.fits"),"psf_$b.fits",force=true )
        run(`modhead psf_$b.fits SAMPLING 0.5`) ## required to inform SE++
    elseif occursin("nir",b)
        fsymlink("../NIR_F0/psf_$(string(b[1])).fits", "psf_$b.fits")
        run(`modhead psf_$b.fits SAMPLING 0.333333`) ## required to inform SE++
    end
    ## Derive background and gain (from affine relation between science and rms images)
    ## new* images contains "ideal" (bg noise free) images derived from weight maps
    b,g = tweak_rms_ima( lima, lwima, "new_"*lima, "new_"*lwima )
    push!( bg_v, b )
    push!( gain_v, g )    
end


open("bg_gain.log","w") do io
    println(io,join(bands," "^4))
    println(io,join(bg_v,' '))
    println(io,join(gain_v,' '))
end


### Chop a smaller chunk of all images to run fast tests! Must "uncomment"
#=
    map( f->run( `imcopy $f[1:2999,1:2999]  sub_$f`),
         filter( x-> occursin("lsst",x) & !occursin("psf",x) & occursin("native",x) , readdir() ) )
    map( f->run( `imcopy $f[1:1999,1:1999]  sub_$f`),
         filter( x-> occursin("nir",x) & !occursin("psf",x) & occursin("native",x) , readdir() ) )
    map( f->run( `imcopy $f[1:5999,1:5999]  sub_$f`),
         filter( x-> occursin("vis",x) & !occursin("psf",x), readdir() ) )
=#

### Add BAND keywords to all fits files, excluding psf files
map( x-> run(`modhead $x BAND $(replace(x,r".*ds_(\w+)\..*"=>s"\1"))`), filter( x->endswith(x,".fits") & !occursin("psf",x) , readdir())  )
