#!/bin/bash

dir=stats
mkdir -p img

########################################
echo "plotting"

timetype=day

colorize() {
    case $1 in
        US) printf "#3333cc" ;;
        GB) printf "#7700cc" ;;
        CA) printf "#ff55ff" ;;
        AU) printf "#7755cc" ;;
        IE) printf "#000099" ;;

        FR) printf "#cc5555" ;;
        BE) printf "#cc3377" ;;
        ES) printf "#cc0033" ;;
        CH) printf "#ee4400" ;;
        PT) printf "#cc5500" ;;
        IT) printf "#005544" ;;

        PH) printf "#cc7777" ;;
        IN) printf "#cc7722" ;;
        CN) printf "#ff0010" ;;
        TW) printf "#ff0055" ;;
        JP) printf "#cc0510" ;;
        MY) printf "#bb6699" ;;
        HK) printf "#ff5555" ;;
        SG) printf "#dd7733" ;;
        TH) printf "#dd3355" ;;
        KR) printf "#dd2222" ;;
        RU) printf "#aa2222" ;;
        ID) printf "#ef7355" ;;

        ZA) printf "#aa5511" ;;
        NG) printf "#aa9955" ;;
        MA) printf "#997799" ;;
        CM) printf "#5577aa" ;;
        CI) printf "#bb5577" ;;
        SA) printf "#555500" ;;
        KW) printf "#888800" ;;
        EG) printf "#cc9944" ;;
        AE) printf "#99cc55" ;;
        OM) printf "#aa8833" ;;
        QA) printf "#99aa00" ;;
        BH) printf "#99aa77" ;;
        IQ) printf "#cccc44" ;;

        BR) printf "#00dd33" ;;
        AR) printf "#005599" ;;
        MX) printf "#996600" ;;
        CO) printf "#00aa44" ;;
        CL) printf "#ccee66" ;;
        UY) printf "#007777" ;;
        VE) printf "#77ac22" ;;
        EC) printf "#88ff66" ;;

        en) printf "#000077" ;;
        fr) printf "#007777" ;;
        es) printf "#770077" ;;
        *)  printf "#000000" ;;
    esac
}

name() {
    case $1 in
        US) printf "USA" ;;
        GB) printf "UK" ;;
        CA) printf "Canada" ;;
        AU) printf "Australia" ;;
        IE) printf "Ireland" ;;

        FR) printf "France" ;;
        BE) printf "Belgium" ;;
        ES) printf "Spain" ;;
        CH) printf "Switzerland" ;;
        PT) printf "Portugal" ;;
        IT) printf "Italy" ;;

        PH) printf "Philippines" ;;
        IN) printf "India" ;;
        CN) printf "China" ;;
        TW) printf "Taiwan" ;;
        JP) printf "Japan" ;;
        MY) printf "Malasia" ;;
        HK) printf "Hong~Kong" ;;
        SG) printf "Singapore" ;;
        TH) printf "Thailand" ;;
        KR) printf "South~Korea" ;;
        RU) printf "Russia" ;;
        ID) printf "Indonesia" ;;

        ZA) printf "South~Africa" ;;
        NG) printf "Nigeria" ;;
        MA) printf "Morocco" ;;
        CM) printf "Cameroon" ;;
        CI) printf "Cote~d'Ivoire" ;;
        SA) printf "Saudi~Arabia" ;;
        KW) printf "Kuwait" ;;
        EG) printf "Egypt" ;;
        AE) printf "Arab~Emirates" ;;
        OM) printf "Oman" ;;
        QA) printf "Qatar" ;;
        BH) printf "Bahrain" ;;
        IQ) printf "Iraq" ;;

        BR) printf "Brazil" ;;
        AR) printf "Argentina" ;;
        MX) printf "Mexico" ;;
        CO) printf "Colombia" ;;
        CL) printf "Chile" ;;
        UY) printf "Uruguay" ;;
        VE) printf "Venezuela" ;;
        EC) printf "Ecuador" ;;

        en) printf "English" ;;
        pt) printf "Portuguese" ;;
        zh) printf "Chinese" ;;
        fr) printf "French" ;;
        ar) printf "Arabic" ;;
        es) printf "Spanish" ;;
        ja) printf "Japanese" ;;
        tr) printf "Turkish" ;;
        in) printf "Indonesian" ;;
        tl) printf "Tagalog" ;;
        ko) printf "Korean" ;;
        de) printf "German" ;;
        nl) printf "Dutch" ;;
        cs) printf "Czech" ;;
        da) printf "Danish" ;;
        el) printf "Greek" ;;
        pl) printf "Polish" ;;
        ru) printf "Russian" ;;
        vi) printf "Vietnamese" ;;
        th) printf "Thai" ;;
        *)  printf $1 ;;
    esac
}

mksorted() {
    tmp=$(mktemp)
    numcountries=10
    sort "$1" -k2 -nr | head -n${numcountries} > "$tmp"
    printf "" > ${1}.sorted
    cat $tmp | while read line; do
        code=$(echo $line | cut -d' ' -f1)
        echo "\\\\scriptsize~$(name $code) $(colorize $code | sed 's/#/0x/') $line" >> ${1}.sorted
    done
    tot_good=$(cut -d' ' -f2 "$tmp" | awk '{s+=$1}END{print s}')
    tot=$(head -n1 "$1" | cut -d' ' -f3)
    per=$(($tot - $tot_good))
    other=$(bc -l <<< "$per / $tot")
    #cat $tmp > ${1}.sorted
    echo "\\\\scriptsize~other 0x333333 other $per $tot $other" >> "${1}.sorted"
}

xdiv=''
stackplots() {
    #printf "'$1' using (column('hr')+0.5):($(expand ${@:2})/column('total')) with boxes"
    printf "'$1' using (column('$timetype')$xdiv):(("
    for arg in ${@:2}; do
        printf "column('$arg')+"
    done
    printf "0)/column('total')) smooth csplines with filledcurve x1 lc rgb '$(colorize $2)'"
    #printf "0)/column('total')) with boxes lc rgb '$(colorize $2)'"
    if [[ -z $3 ]]; then
        echo
    else
        echo ',\'
        stackplots "$1" ${@:3}
    fi
}

mkplots() {
    lang=$1
    echo "  mkplots $lang"
    info_time=$dir/lang_${timetype}_country/${lang}.dat
    info_country=$dir/lang/${lang}.dat
    mksorted "$info_country"
    topk=$(cut -d' ' -f3 ${info_country}.sorted | tac | sed s/other//)
    #return
    if [[ $timetype = hr ]]; then
        xmax=24
        xstep=3
        xformat="\\\\scriptsize %g:00"
        xlabel="time of day (UTC)"
        xval="column('$timetype')"
        xdiv=''
    else
        xmax=7
        xstep=1
        xformat="\\\\scriptsize %g"
        xlabel="day of week(UTC)"
        xval="column('$timetype')/24"
        xdiv='/24'
    fi
gnuplot << EOF

set terminal epslatex size 7in,1.75in
set output "img/${timetype}-${lang}.tex

set style fill solid 0.5
set multiplot layout 1,2

unset key
set xrange [0:$xmax]
set xtics $xstep
set format x "$xformat"
set yrange [0:1]
set tics front nomirror
unset ytics
set xlabel "\\\\scriptsize $xlabel" offset 0,0.5
set ylabel "\\\\scriptsize fraction of tweets"
set size 0.75,1
set title 'Countries that Tweet in $(name $lang)' offset 0,-0.5
plot '$info_time' using ($xval):(1) with filledcurve x1 lc rgb '#333333',\
$(stackplots "$info_time" $topk)

set title ' '
set xtics autofreq
unset xtics
set xtics " " 1
set autoscale x
set autoscale y
unset ylabel
set border 0
set style line 101 lc rgb '#808080' lt 1 lw 1
unset xlabel
#set format x ''
#set format y ''
set tics scale 0
set size 0.25,1
set origin 0.75,0
set lmargin 7
set xlabel '~~' #offset -5
plot '${info_country}.sorted' using (\$4):(column(0)):(\$4):(0.4):2:ytic(1) with boxxyerrorbars lc rgb variable

EOF
}

for t in day hr; do
    timetype=$t
    mkplots en
    mkplots es
    mkplots fr
    mkplots ar
    mkplots zh
    mkplots pt
    mkplots ko
    mkplots ja
    mkplots tr
    mkplots tl
    mkplots de
    mkplots nl
    mkplots cs
    mkplots da
    mkplots 'in'
    mkplots und
    mkplots de
    mkplots nl
    mkplots cs
    mkplots da
    mkplots el
    mkplots pl
    mkplots ru
    mkplots vi
    mkplots th
done

################################################################################

echo "country/lang"

mksorted "$dir/lang.dat"
mksorted "$dir/country.dat"

gnuplot << EOF
set terminal epslatex size 3.5in,2.25in
unset key
set ylabel "\\\\scriptsize fraction of tweets" offset 3
set style fill solid 0.4
set boxwidth 0.8
set xtics rotate by -45
set format y "\\\\scriptsize %0.1f"
set ytics 0.1 nomirror
set xtics nomirror
set border 3
set lmargin 3
set bmargin 3

set output "img/country.tex
set title "Most Active Countries" offset 0,-2
plot "$dir/country.dat.sorted" using 6:xtic(1) with boxes lc rgb '#ff0000'
#plot "$dir/country.dat.sorted" using 0:6:2:xtic(1) with boxes lc rgb variable

set output "img/lang.tex"
set title "Most Active Languages" offset 0,-2
plot "$dir/lang.dat.sorted" using 6:xtic(1) with boxes lc rgb '#ff0000'
#plot "$dir/lang.dat.sorted" using 0:6:2:xtic(1) with boxes lc rgb variable
EOF
