# using Pkg ; Pkg.add.( ["FITSIO", "CFITSIO", "Plots", "NearestNeighbors", "StatsBase"] )
using Plots, NearestNeighbors, FITSIO, CFITSIO, StatsBase, Printf

##############################
function catal_merge( fn1::AbstractString, fn2::AbstractString ;
                      cn1=("X","Y"), cn2=("x","y"), max_sep::Real=6.0,
                      leafsize::Integer=10)
    io=FITS(fn1)
    io2=FITS(fn2)
    gt_pos = permutedims([read(io[2],cn1[1]) read(io[2],cn1[2])] )
    n1=size(gt_pos,2)
    tr = KDTree( permutedims([read(io2[2],cn2[1]) read(io2[2],cn2[2])]);
                 leafsize=leafsize )
    idxs, dists = nn(tr, gt_pos)
    ok = findall( dists .< max_sep )  # in pixels
    @info "in catalogue $fn1 (ground truth):  $(length(idxs)) entries, of which $(length(ok)) have a close match in $fn2"
    res=Dict{String,Any}()
    ## column names in first gt file are tweaked to avoid ambiguities
    map( c->( res["gt_"*c] = read(io[2],c)[ok] ), FITSIO.colnames(io[2]) ) 
    close(io)
    map( c->( res[c] = read(io2[2],c)[idxs][ok] ), FITSIO.colnames(io2[2]) )
    close(io2)
    res["separation"] = dists[ok]
    res
end

linsigmoid(x,r) = log( (x-r[1])/(r[2]-x))
linsigmoidinv(y,r) = (r[1]+r[2]*exp(y))/(1+exp(y))
logsigmoid(x,r) = linsigmoidinv( log(x), log.(r) )
logsigmoidinv(y,r) = exp( linsigmoidinv( log(x), log.(r) ) )
linsigmoid(x::T) where {T<:Real} = linsigmoid(x,(T(-0.01),T(1.01)))
logsigmoid(x::T) where {T<:Real} = logsigmoid(x,(T(-0.01),T(1.01)))
linsigmoidinv(y::T) where {T<:Real} = linsigmoidinv(y,(T(-0.01),T(1.01)))
logsigmoidinv(y::T) where {T<:Real} = logsigmoidinv(y,(T(-0.01),T(1.01)))



########################################
function running_average( x::Vector, y::Vector; nbins=30, xrange=nothing, corrected=false )
    ( xrange == nothing ) &&  ( xrange=extrema(x) )
    xbound = collect(LinRange(xrange[1],xrange[2],nbins+1))
    dx = xbound[2] - xbound[1]
    ym = zeros( eltype(x[1]), nbins)
    nc = zeros( Int64, nbins)
    ymed = zeros( eltype(x[1]), nbins)
    ydev = zeros( eltype(x[1]), nbins)
    ynmad = zeros( eltype(x[1]), nbins)
    for j=1:nbins
        ok = findall( xbound[j] .<= x .< xbound[j+1] )
        nc[j] =length(ok)
        if nc[j]>0
            ( ym[j], ydev[j] ) = StatsBase.mean_and_std( y[ok], corrected=corrected )
            ymed[j] = median( y[ok] )
            ynmad[j] = mad( y[ok], normalize=true )
        else
            ym[j]=NaN
            ymed[j]=NaN
            ydev[j]=NaN
            ynmad[j]=NaN
        end
    end
    xbound = xbound[1:end-1] .+ dx/2
    return (xbound,ym,ymed,ydev,ynmad,nc)
end


isdefined(Main,:res) || ( res=catal_merge( "ggg.fits", "fair_vis.cat" ) )
ph_k = replace.( filter( x->occursin(r"^gt_",x) & occursin( r"_bt$",x),  collect(keys(res))), r"(gt_.*)_bt"=>s"\1")
stt = map( x->summarystats( linsigmoid.( res[x*"_bt"])), ph_k)
medv = [round(x.median,digits=2) for x in stt]
ph_k = ph_k[sortperm(medv)]
wid = [ round( x.q75-x.q25 ,digits=2) for x in stt ][sortperm(medv)]
sort!( medv )

dang = mod.( rad2deg.(res["angle"]) .- (res["gt_disk_angle"]) .+90,180) .-  90 
dbmag = res["mag_vis"] .- 2.5*log10.( res["bt_vis"] ) .-
    ( res["gt_VIS"] .- 2.5*log10.( res["gt_VIS_bt"] ) )
ddmag = res["mag_vis"] .- 2.5*log10.( 1 .- res["bt_vis"] ) .-
    ( res["gt_VIS"] .- 2.5*log10.( 1 .- res["gt_VIS_bt"] ) )
ddr = log10.(res["disk_effR_px"]/10) .- log10.(res["gt_disk_effR"])
dbr = log10.(res["bulge_effR_px"]/10) .- log10.(res["gt_bulge_effR"])

########################################
function myplot( y, yn, yl; x=res["gt_VIS"], xn="True VIS", xl=(16,26.7) )
    rav = running_average( x, y, nbins=2, xrange=(21.5,22.5) )
    s = rav[3][1] ; ds = rav[4][1] / sqrt( rav[6][1] )
    p=plot( ; xlabel=xn, legend=:topleft, framestyle=:box, xlims=xl, link=:x)
    scatter!( p , x, y, ylims=yl, ms=3, msw=-1, ma=0.5, label="" )
    str=@sprintf("%s .  bias: %.2e pm %.2e",yn,s,ds)
    rav = running_average( x, y, nbins=12, xrange=(18,26) )
    plot!(p, rav[1], rav[3], ribbon=rav[5] / sqrt.(rav[6]), label=deepcopy(str), lw=4 )
    p
end
########################################
function myplot_scat( y, dy, yn, yl; x=res["gt_VIS"], xn="True VIS", xl=(16,26.7) )
#    s = rav[3][1] ; ds = rav[4][1] / sqrt( rav[6][1] )
    p=plot( ; xlabel=xn, legend=:topleft, framestyle=:box, xlims=xl, link=:x, title=yn )
    scatter!( p, x, dy, ylims=yl, ms=3, msw=-1, ma=0.5, label="", yscale=:log10 )
    rav = running_average( x, y, nbins=12, xrange=(18,26) )
    rav2 = running_average( x, dy, nbins=12, xrange=(18,26) )
    s = median( filter( x->!isnan(x) , rav[5] ./ rav2[3] ) )
    str=@sprintf("%.2f", s)
    plot!(p, rav[1], rav[5], lw=4, label="NMAD scatter" )
    plot!(p, rav2[1], rav2[3], lw=4, label="MED error~NMAD/$str" )
    p
end

plist_bias=[]
push!( plist_bias,
       myplot( dang, "Delta angle", (-20,20) ) )
push!( plist_bias,
       myplot( res["disk_axr"] .- res["gt_disk_axr"], "Disk Axr", (-0.2,0.2) ) )
push!( plist_bias,
       myplot( res["bulge_axr"] .- res["gt_bulge_axr"], "Bulge Axr", (-0.2,0.2) ) )
push!( plist_bias,
       myplot( res["mag_vis"] .- res["gt_VIS"], "Total VIS mag", (-0.5,0.5) ) )
push!( plist_bias,
       myplot( log10.(res["bt_vis"]) .- log10.(res["gt_VIS_bt"]), "VIS log10 BoT", (-0.5,0.5) ) )
push!( plist_bias,
       myplot( ddr, "VIS log10 Rdisk", (-1,1) ) )
push!( plist_bias,
       myplot( dbr, "VIS log10 Rbulge",(-1,1) ) )
dgmi= res["mag_g_lsst"] .- res["mag_i_lsst"]  .-
    ( res["gt_LSST_G"] .- res["gt_LSST_I"] )
dymh= res["mag_y_nir"] .- res["mag_h_nir"]  .-
    ( res["gt_NIR_Y"] .- res["gt_NIR_H"] )
push!( plist_bias,
       myplot( dgmi, "g-i total color", (-0.4,0.4) ) )
push!( plist_bias,
       myplot( dymh, "Y-H total color",(-0.4,0.4) ) )
plot( plist_bias..., link=:x )


plist_scatter=[]

push!( plist_scatter,
        myplot_scat( dang, rad2deg.(res["angle_err"]) , "Delta angle", (0.01,40) ) )
push!( plist_scatter,
       myplot_scat( res["disk_axr"] .- res["gt_disk_axr"], res["disk_axr_err"],  "Disk Axr", (0.001,1) ) )
#push!( plist_scatter,
#       myplot_scat( res["bulge_axr"] .- res["gt_bulge_axr"], res["bulge_axr_err"],  "Bulge Axr", (0.001,1) ) )
push!( plist_scatter,
       myplot_scat( res["mag_vis"] .- res["gt_VIS"], res["mag_vis_err"], "Total VIS mag", (0.001,1) ) )
push!( plist_scatter,
       myplot_scat( res["bt_vis"] .- res["gt_VIS_bt"], res["bt_vis_err"], "VIS BoT", (0.001,1) ) )
#=
push!( plist_scatter,
       myplot_scat( ddr, 2.3 * res["disk_effR_px_err"] ./ res["disk_effR_px"], "VIS log10 Rdisk", (0.001,1) ) )
push!( plist_scatter,
       myplot_scat( ddr, 2.3 * res["bulge_effR_px_err"] ./ res["bulge_effR_px"], "VIS log10 Rbulge", (0.001,1) ) )

dgmi= res["mag_g_lsst"] .- res["mag_i_lsst"]  .-
    ( res["gt_LSST_G"] .- res["gt_LSST_I"] )
dymh= res["mag_y_nir"] .- res["mag_h_nir"]  .-
    ( res["gt_NIR_Y"] .- res["gt_NIR_H"] )
push!( plist_bias,
       myplot( dgmi, "g-i total color", (-0.4,0.4) ) )
push!( plist_bias,
       myplot( dymh, "Y-H total color",(-0.4,0.4) ) )
=#
plot( plist_scatter..., link=:x )


#=
rav  = running_average( res["gt_VIS"], dang , nbins=12, xrange=(18,26) )
rav2 = running_average( res["gt_VIS"], rad2deg.(res["angle_err"]), nbins=12, xrange=(18,26))
scatter( res["gt_VIS"],  rad2deg.(res["angle_err"]) , ms=3, msw=-1, ylims=(0.02,20), ma=0.4 , xlims=(16,26.7) , yscale=:log10 , legend=:topleft)
plot!( rav[1], rav[5], lw=4, label="NMAD scatter" )
plot!( rav2[1], rav2[3], lw=4, label="MED error" )
=#
