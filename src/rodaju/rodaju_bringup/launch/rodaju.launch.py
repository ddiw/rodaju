from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_voice = LaunchConfiguration('use_voice')
    use_ui = LaunchConfiguration('use_ui')

    return LaunchDescription([
        # ── 실행 인자 ──────────────────────────────────────────────
        DeclareLaunchArgument(
            'use_voice',
            default_value='true',
            description='음성 명령 노드 실행 여부',
        ),
        DeclareLaunchArgument(
            'use_ui',
            default_value='true',
            description='웹 UI 노드 실행 여부',
        ),

        # ── Vision Node ────────────────────────────────────────────
        Node(
            package='vision_node',
            executable='vision',
            name='vision_node',
            output='screen',
            parameters=[{
                'publish_rate': 10.0,
                'conf_threshold': 0.70,
            }],
        ),

        # ── Manager Node ───────────────────────────────────────────
        Node(
            package='manager_node',
            executable='manager',
            name='manager_node',
            output='screen',
        ),

        # ── Execute Node ───────────────────────────────────────────
        Node(
            package='execute_node',
            executable='execute',
            name='execute_node',
            output='screen',
        ),

        # ── Voice Command Node (선택) ──────────────────────────────
        Node(
            package='voice_command_node',
            executable='voice_command_node',
            name='voice_command_node',
            output='screen',
            condition=IfCondition(use_voice),
        ),

        # ── UI Node (선택) ─────────────────────────────────────────
        Node(
            package='UI_node',
            executable='ui_node',
            name='ui_node',
            output='screen',
            condition=IfCondition(use_ui),
        ),
    ])
