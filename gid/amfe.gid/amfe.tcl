#################################################
#      GiD-Tcl procedures invoked by GiD        #
#################################################
proc InitGIDProject { dir } {
    AMfe::SetDir $dir ;#store to use it later
}

proc AfterWriteCalcFileGIDProject { filename errorflag } {
    if { ![info exists gid_groups_conds::doc] } {
        WarnWin [= "Error: data not OK"]
        return
    }
    set err [catch { AMfe::WriteAMfeMesh $filename } ret]
    if { $err } {
        WarnWin [= "Error when preparing data for analysis (%s)" $::errorInfo]
        set ret -cancel-
    }
    return $ret
}

#################################################
#      namespace implementing procedures        #
#################################################
namespace eval AMfe {
    variable problemtype_dir
}

proc AMfe::SetDir { dir } {
    variable problemtype_dir
    set problemtype_dir $dir
}

proc AMfe::GetDir { } {
    variable problemtype_dir
    return $problemtype_dir
}



##################################################
# Write AMfe Mesh
proc AMfe::WriteAMfeMesh { filename } {
    set fp [GiD_File fopen $filename w]
    GiD_File fprintf $fp %s "\{"
    GiD_File fprintf $fp {  %s %s%s} \"version\": 0.1 ,
    GiD_File fprintf $fp {  %s %s%s%s%s} \"filename\": \" [GiD_Info Project ModelName] \" ,
    GiD_File fprintf $fp {  %s %s%s} \"quadratic\": [GiD_Info Project Quadratic] ,
    GiD_File fprintf $fp {  %s %s%s} \"element_size\": [GiD_Info Project LastElementSize] ,
    GiD_File fprintf $fp {  %s %d%s} \"no_of_nodes\": [GiD_Info Mesh NumNodes] ,
    GiD_File fprintf $fp {  %s %d%s} \"no_of_elements\": [GiD_Info Mesh NumElements Any] ,
    GiD_File fprintf $fp {  %s} \"nodes\":
    GiD_File fprintf $fp {  %s} \[
    set nodes [GiD_Info Mesh Nodes -sublist]
    set index 0
    foreach node $nodes {
        if { $index > 0 } {
            GiD_File fprintf -nonewline $fp {%s} ",\n    "
        } else {
            GiD_File fprintf -nonewline $fp {%s} "    "
        }
        GiD_File fprintf -nonewline $fp {%s %10d %s %s %s %16.9e %s %16.9e %s %16.9e %s } \{"id\": [lindex $node 0] , \"coords\": \[ [lindex $node 1] , [lindex $node 2] , [lindex $node 3] \]\}
        incr index
    }
    GiD_File fprintf $fp {%s} "\n  \],"
    GiD_File fprintf $fp {  %s} \"elements\":
    GiD_File fprintf -nonewline $fp {  %s} \[
    set eletypes [list Line Triangle Quadrilateral Tetrahedra Prism Pyramid Point Sphere Circle]
    set eletypeindex 0
    set one_eletypeset_written 0
    foreach eletype $eletypes {
        set no_of_elements [GiD_Info Mesh NumElements $eletype]
        if {$no_of_elements > 0} {
            if { $one_eletypeset_written == 1 } {
                GiD_File fprintf $fp {%s} ",\n    \{"
            } else {
                GiD_File fprintf $fp {%s} "\n    \{"
                set one_eletypeset_written 1
            }
            GiD_File fprintf $fp {    %s %s%s%s} \"ele_type\": \" $eletype \",
            GiD_File fprintf $fp {    %s %d%s} \"number\": $no_of_elements ,
            GiD_File fprintf $fp {    %s} \"elements\":
            GiD_File fprintf -nonewline $fp {      %s} \[
            set elements [GiD_Info Mesh Elements $eletype -sublist]
            set eleindex 0
            foreach element $elements {
                if {$eleindex == 0} {
                    GiD_File fprintf -nonewline $fp {%s} "\n        \{"
                } else {
                    GiD_File fprintf -nonewline $fp {%s} ",\n        \{"
                }
                incr eleindex
                set index 0
                foreach entry $element {
                    if {$index == 0} {
                        GiD_File fprintf $fp {%s%s %10d%s} "\n          " \"id\": $entry ,
                        GiD_File fprintf -nonewline $fp {          %s %s} \"connectivity\": \[
                        incr index
                    } elseif {$index == 1} {
                        GiD_File fprintf -nonewline $fp {%10d} $entry
                        incr index
                    } else {
                        GiD_File fprintf -nonewline $fp {%s %10d} , $entry
                        incr index
                    }
                }
                GiD_File fprintf -nonewline $fp {%s} "\]\n        \}"
            }
            GiD_File fprintf $fp {%s} "\n      \]"
            GiD_File fprintf -nonewline $fp {    %s} \}
        }
        incr eletypeindex
    }
    GiD_File fprintf $fp {%s} "\n  \],"
    GiD_File fprintf $fp {  %s} \"groups\":
    GiD_File fprintf -nonewline $fp {    %s} \[
    set group_names [GiD_Groups list]
    set groupindex 0
    foreach group $group_names {
        if {$groupindex == 0} {
            GiD_File fprintf $fp {%s} "\n      \{"
        } else {
            GiD_File fprintf $fp {%s} ",\n      \{"
        }
        incr groupindex
        GiD_File fprintf $fp {        %s %s%s%s} \"name\": \" $group \",
        set nodes [lindex [GiD_EntitiesGroups get $group all_mesh] 0]
        set elements [lindex [GiD_EntitiesGroups get $group all_mesh] 1]
        GiD_File fprintf -nonewline $fp {        %s %s} \"nodes\": \[
        set nodesindex 0
        foreach node $nodes {
            if {$nodesindex == 0} {
                GiD_File fprintf -nonewline $fp {%s} $node
            } else {
                GiD_File fprintf -nonewline $fp {%s %s} , $node
            }
            incr nodesindex
        }
        GiD_File fprintf $fp {%s} \],
        GiD_File fprintf -nonewline $fp {        %s %s} \"elements\": \[
        set elementsindex 0
        foreach element $elements {
            if {$elementsindex == 0 } {
                GiD_File fprintf -nonewline $fp {%s} $element
            } else {
                GiD_File fprintf -nonewline $fp {%s %s} , $element
            }
            incr elementsindex
        }
        GiD_File fprintf $fp {%s} \]
        GiD_File fprintf -nonewline $fp {      %s} \}
    }
    GiD_File fprintf $fp {%s} "\n    \]"
    GiD_File fprintf $fp {%s} \}
    GiD_File fclose $fp
}

