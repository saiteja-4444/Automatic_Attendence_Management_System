/*
SQLyog Enterprise - MySQL GUI v6.56
MySQL - 5.5.5-10.1.13-MariaDB : Database - smart_attendance
*********************************************************************
*/

/*!40101 SET NAMES utf8 */;

/*!40101 SET SQL_MODE=''*/;

/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;

CREATE DATABASE /*!32312 IF NOT EXISTS*/`smart_attendance` /*!40100 DEFAULT CHARACTER SET latin1 */;

USE `smart_attendance`;

/*Table structure for table `attendance` */

DROP TABLE IF EXISTS `attendance`;

CREATE TABLE `attendance` (
  `id` int(10) NOT NULL AUTO_INCREMENT,
  `rno` varchar(100) DEFAULT NULL,
  `semail` varbinary(100) DEFAULT NULL,
  `in_time` varchar(100) DEFAULT NULL,
  `in_status` varchar(100) DEFAULT NULL,
  `out_time` varchar(100) DEFAULT NULL,
  `out_status` varchar(100) DEFAULT NULL,
  `date1` varchar(100) DEFAULT NULL,
  `m1` varchar(100) DEFAULT NULL,
  `punches` int(100) DEFAULT '0',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=4 DEFAULT CHARSET=latin1;

/*Data for the table `attendance` */

/*Table structure for table `mid_results` */

DROP TABLE IF EXISTS `mid_results`;

CREATE TABLE `mid_results` (
  `id` int(10) NOT NULL AUTO_INCREMENT,
  `sname` varchar(100) DEFAULT NULL,
  `semail` varchar(100) DEFAULT NULL,
  `rno` varchar(100) DEFAULT NULL,
  `pno` varchar(100) DEFAULT NULL,
  `year` varchar(100) DEFAULT NULL,
  `sem` varchar(100) DEFAULT NULL,
  `section` varchar(100) DEFAULT NULL,
  `subject` varchar(100) DEFAULT NULL,
  `marks` varchar(30) DEFAULT NULL,
  `d1` varchar(100) DEFAULT NULL,
  `mcomp` varchar(100) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=21 DEFAULT CHARSET=latin1;

/*Data for the table `mid_results` */

insert  into `mid_results`(`id`,`sname`,`semail`,`rno`,`pno`,`year`,`sem`,`section`,`subject`,`marks`,`d1`,`mcomp`) values (10,'Lakshmi','vamsi@ymtsindia.in','2','9705920956','1','1-Sem','Mid-1','m1','1','27-11-2024','11-SemMid-1'),(11,'Lakshmi','vamsi@ymtsindia.in','2','9705920956','1','1-Sem','Mid-1','m2','10','27-11-2024','11-SemMid-1'),(12,'Lakshmi','vamsi@ymtsindia.in','2','9705920956','1','1-Sem','Mid-1','m3','3','27-11-2024','11-SemMid-1'),(13,'Lakshmi','jaswanthmarasu@gmail.com','2','9705920956','2','2-Sem','Mid-2','Telugu','45','27-11-2024','22-SemMid-2'),(14,'Lakshmi','jaswanthmarasu@gmail.com','2','9705920956','2','2-Sem','Mid-2','English','30','27-11-2024','22-SemMid-2'),(15,'Lakshmi','cse.takeoff@gmail.com','2','9705920956','4','1-Sem','Mid-1','A1','20','27-11-2024','41-SemMid-1'),(16,'Lakshmi','cse.takeoff@gmail.com','2','9705920956','4','1-Sem','Mid-1','B1','30','27-11-2024','41-SemMid-1'),(17,'Lakshmi','cse.takeoff@gmail.com','2','9705920956','3','2-Sem','Mid-2','OS','10','27-11-2024','32-SemMid-2'),(18,'Lakshmi','cse.takeoff@gmail.com','2','9705920956','3','2-Sem','Mid-2','DS','20','27-11-2024','32-SemMid-2'),(19,'Lakshmi','cse.takeoff@gmail.com','2','9705920956','1','2-Sem','Mid-2','HTML','45','27-11-2024','12-SemMid-2'),(20,'Lakshmi','cse.takeoff@gmail.com','2','9705920956','1','2-Sem','Mid-2','CSS','20','27-11-2024','12-SemMid-2');

/*Table structure for table `punches` */

DROP TABLE IF EXISTS `punches`;

CREATE TABLE `punches` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `roll_no` varchar(11) DEFAULT NULL,
  `timing` varchar(50) DEFAULT NULL,
  `status` varchar(40) DEFAULT NULL,
  `date` varchar(10) DEFAULT NULL,
  `month` varchar(10) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=14 DEFAULT CHARSET=latin1;

/*Data for the table `punches` */

insert  into `punches`(`id`,`roll_no`,`timing`,`status`,`date`,`month`) values (2,'2','10:53:36','Late In','23-11-2024','November'),(3,'2','12:16:59','Lunch Out','23-11-2024','November'),(7,'2','12:52:18','Lunch In','23-11-2024','November'),(8,'2','16:53:42','Day Out','23-11-2024','November'),(9,'2','09:58:12','Early In','25-11-2024','November'),(10,'2','12:00:07','Lunch Out','25-11-2024','November'),(12,'2','12:53:14','Lunch In','25-11-2024','November'),(13,'2','04:10:01','Day Out','25-11-2024','November');

/*Table structure for table `students` */

DROP TABLE IF EXISTS `students`;

CREATE TABLE `students` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(100) DEFAULT NULL,
  `email` varchar(100) DEFAULT NULL,
  `phone` varchar(100) DEFAULT NULL,
  `roll_no` varchar(11) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=4 DEFAULT CHARSET=latin1;

/*Data for the table `students` */

insert  into `students`(`id`,`name`,`email`,`phone`,`roll_no`) values (3,'Lakshmi','cse.takeoff@gmail.com','9705920956','2');

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
