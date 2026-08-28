# launch/sysid.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    # ----- CLI args -----
    wamv_arg    = DeclareLaunchArgument('wamv', default_value='asv')
    test_arg    = DeclareLaunchArgument('test_type', default_value='bollard_pull')
    port_arg    = DeclareLaunchArgument('port_cmd', default_value='100')
    stbd_arg    = DeclareLaunchArgument('stbd_cmd', default_value='100')
    sim_arg     = DeclareLaunchArgument('Simulation', default_value='false')
    sim_max_arg = DeclareLaunchArgument('sim_thrust_max_n', default_value='1000.0')

    # Substitutions
    wamv = LaunchConfiguration('wamv')

    # ----- system_identification node -----
    sysid_params = {
        'wamv':             wamv,
        'test_type':        LaunchConfiguration('test_type'),
        'port_cmd':         ParameterValue(LaunchConfiguration('port_cmd'), value_type=int),
        'stbd_cmd':         ParameterValue(LaunchConfiguration('stbd_cmd'), value_type=int),
        'Simulation':       ParameterValue(LaunchConfiguration('Simulation'), value_type=bool),
        'sim_thrust_max_n': ParameterValue(LaunchConfiguration('sim_thrust_max_n'), value_type=float),
    }

    sysid_node = Node(
        package='system_identification',
        executable='system_identification',   # matches your setup.py entry point
        name='system_identification',
        output='screen',
        parameters=[sysid_params],
    )

    return LaunchDescription([
        wamv_arg, test_arg, port_arg, stbd_arg, sim_arg, sim_max_arg,
        sysid_node,
    ])
